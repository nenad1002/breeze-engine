"""Finalize a LOCAL Qwen3.5 bundle. Decoder export/quantization is NOT performed."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import tempfile

from breeze.qwen35_config import QwenConfig


METADATA = ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
            "generation_config.json", "special_tokens_map.json", "added_tokens.json",
            "vocab.json", "merges.txt", "tokenizer.model")


def _publish(temporary, output, overwrite=False):
    if overwrite:
        os.replace(temporary, output)
    else:
        # Atomic publication without replacing a concurrently created destination.
        os.link(temporary, output)


def _temporary(output):
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    os.close(fd)
    return Path(name)


def _embedding_tensor(source):
    """Locate exactly one local language embedding tensor, including sharded checkpoints."""
    from safetensors import safe_open

    index = source / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text())["weight_map"]
        matches = [(name, source / shard) for name, shard in weight_map.items()
                   if name.endswith("embed_tokens.weight") and "vision" not in name]
    else:
        matches = []
        for shard in sorted(source.glob("*.safetensors")):
            with safe_open(shard, framework="numpy") as handle:
                matches.extend((name, shard) for name in handle.keys()
                               if name.endswith("embed_tokens.weight") and "vision" not in name)
    if len(matches) != 1:
        raise ValueError(f"Expected one language embedding tensor, found {len(matches)}")
    name, shard = matches[0]
    # Index entries are checkpoint-relative; checkpoint files themselves may be local symlinks.
    if shard.is_absolute() and not shard.absolute().is_relative_to(source.absolute()):
        raise ValueError("Embedding shard must be inside the local source folder")
    if ".." in shard.relative_to(source).parts or not shard.is_file():
        raise ValueError("Embedding shard must be an existing local checkpoint file")
    return name, shard


def _bfloat16_view(shard, name, shape):
    """Map BF16 storage without requiring torch or a NumPy BF16 dtype."""
    import numpy as np

    with shard.open("rb") as stream:
        length = int.from_bytes(stream.read(8), "little")
        if not 0 < length <= min(shard.stat().st_size - 8, 100_000_000):
            raise ValueError("Invalid safetensors header length")
        header = json.loads(stream.read(length))
    entry = header[name]
    if entry["dtype"] != "BF16":
        return None
    begin, end = entry["data_offsets"]
    if (tuple(entry["shape"]) != shape or begin < 0 or end - begin != shape[0] * shape[1] * 2
            or 8 + length + end > shard.stat().st_size):
        raise ValueError("Invalid BF16 embedding shape or storage bounds")
    return np.memmap(shard, mode="r", dtype="<u2", offset=8 + length + begin, shape=shape)


def extract_embeddings(source, output, config, *, overwrite=False):
    """Atomically write FP16 embeddings in at most 4096-row blocks, without torch."""
    import numpy as np
    from safetensors import safe_open

    source, output = Path(source).absolute(), Path(output).absolute()
    if output.is_symlink() or any(output.resolve() == p.resolve() for p in source.iterdir() if p.is_file()):
        raise ValueError("Embedding output must not overwrite a source file or symlink")
    if output.exists() and not overwrite:
        raise FileExistsError("Embeddings already exist; use --overwrite explicitly")
    name, shard = _embedding_tensor(source)
    if output.resolve() == shard.resolve():
        raise ValueError("Embedding output must not overwrite its source shard")
    shape = (config.vocab_size, config.hidden_size)
    temporary = None
    try:
        with safe_open(shard, framework="numpy") as handle:
            # BF16 is mapped as integer storage because NumPy need not support that dtype.
            bf16 = _bfloat16_view(shard, name, shape)
            view = None if bf16 is not None else handle.get_slice(name)
            if view is not None and tuple(view.get_shape()) != shape:
                raise ValueError("Embedding shape does not match the text config")
            temporary = _temporary(output)
            array = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float16, shape=shape)
            try:
                for start in range(0, config.vocab_size, 4096):
                    stop = min(start + 4096, config.vocab_size)
                    if bf16 is not None:
                        block = (bf16[start:stop].astype(np.uint32) << 16).view(np.float32)
                    else:
                        block = view[start:stop]
                    array[start:stop] = np.asarray(block, dtype=np.float16)
                array.flush()
            finally:
                del array
        _publish(temporary, output, overwrite)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _copy_metadata(source, output):
    temporary = _temporary(output)
    try:
        shutil.copyfile(source, temporary)
        _publish(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def _write_manifest(output, manifest, overwrite):
    temporary = _temporary(output)
    try:
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        _publish(temporary, output, overwrite)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_decoder(model, config, destinations):
    """Check packed weights and protect every external weight file from replacement."""
    import onnx
    from breeze.loader import load_graph
    from breeze.qwen35_weights import bind_weights

    proto = onnx.load(str(model), load_external_data=False)
    protected = {model.resolve()}
    for tensor in proto.graph.initializer:
        if tensor.data_location == onnx.TensorProto.EXTERNAL:
            info = {item.key: item.value for item in tensor.external_data}
            location = Path(info["location"])
            if location.is_absolute() or ".." in location.parts:
                raise ValueError("Decoder weight location must stay inside --model-dir")
            protected.add((model.parent / location).resolve())
    if any(path.resolve() in protected for path in destinations):
        raise ValueError("A bundle output conflicts with an existing decoder weight file")
    del proto
    graph = load_graph(model, mmap_external=True)
    if "inputs_embeds" not in graph.inputs or "logits" not in graph.outputs:
        raise ValueError("Decoder graph must accept inputs_embeds and produce logits")
    nodes, _ = bind_weights(graph, config)
    return Counter(int(node.attrs["bits"]) for node in nodes.values())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="Local original checkpoint folder containing config, tokenizer and safetensors")
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="Local bundle directory with an ALREADY-PREPARED quantized model.onnx decoder")
    parser.add_argument("--execute", action="store_true", help="Validate, copy local metadata and extract embeddings; default: dry plan")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing embeddings and manifest only; never decoder weights or conflicting metadata")
    args = parser.parse_args(argv)
    source, output = args.source.absolute(), args.model_dir.absolute()
    model, embedding = output / "model.onnx", output / "embeddings.npy"
    manifest_path = output / "breeze_manifest.json"
    try:
        if not source.is_dir() or not output.is_dir():
            raise ValueError("--source and --model-dir must be existing local directories")
        if source.resolve() == output.resolve():
            raise ValueError("--source and --model-dir must be different directories")
        if not model.is_file():
            raise ValueError("--model-dir requires an already-prepared model.onnx; decoder export/quantization is NOT performed")
        if not (source / "tokenizer.json").is_file() or not any(source.glob("*.safetensors")):
            raise ValueError("--source requires tokenizer.json and local safetensors checkpoint files")
        raw_config = json.loads((source / "config.json").read_text())
        config = QwenConfig.from_dict(raw_config)
        metadata = [name for name in METADATA if (source / name).is_file()]
        for path in [embedding, manifest_path, *(output / name for name in metadata)]:
            if path.is_symlink() or path.is_dir():
                raise ValueError(f"Refusing bundle output symlink or directory: {path}")
        for path in (embedding, manifest_path):
            if path.exists() and not args.overwrite:
                raise FileExistsError(f"{path.name} already exists; use --overwrite explicitly")
        to_copy = []
        for name in metadata:
            target = output / name
            if target.resolve() == (source / name).resolve():
                raise ValueError("Metadata output must not overwrite its source")
            if target.exists():
                if target.read_bytes() != (source / name).read_bytes():
                    raise ValueError(f"Existing metadata conflicts with source: {name}; refusing replacement")
            else:
                to_copy.append(name)
        plan = {"source": str(source.resolve()), "model": str(model.resolve()),
                "metadata_to_copy": to_copy, "embeddings": str(embedding),
                "manifest": str(manifest_path), "overwrite": args.overwrite,
                "config": asdict(config)}
        if not args.execute:
            print(json.dumps({**plan, "status": "dry plan; graph validation and extraction pending"}, indent=2))
            print("Decoder export/quantization is NOT performed. Add --execute for local finalization.")
            return plan
        bit_counts = _validate_decoder(model, config, [embedding, manifest_path, *(output / n for n in metadata)])
        # All graph/config/metadata checks precede the potentially large extraction.
        extract_embeddings(source, embedding, config, overwrite=args.overwrite)
        for name in to_copy:
            _copy_metadata(source / name, output / name)
        manifest = {"source_local": str(source.resolve()), "model_local": str(model.resolve()),
                    "source_identity": raw_config.get("_name_or_path") or source.resolve().name,
                    "model_type": raw_config.get("model_type"), "config": asdict(config),
                    "matmuls": sum(bit_counts.values()), "weight_bits": sorted(bit_counts),
                    "weight_bit_counts": dict(sorted(bit_counts.items())), "embedding_dtype": "float16",
                    "status": "structural-only", "validation": "Packed weight shapes checked; no numerical or inference validation"}
        _write_manifest(manifest_path, manifest, args.overwrite)
    except (OSError, ValueError, KeyError, ImportError) as exc:
        parser.error(str(exc))
    print(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    main()