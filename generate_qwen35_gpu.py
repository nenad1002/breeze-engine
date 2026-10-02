"""Standalone greedy generation with the Breeze dense Qwen3.5-9B GPU decoder."""
import argparse
from dataclasses import fields
import json
from pathlib import Path
import time

import numpy as np

from breeze.chat import chat_prompt
from breeze.qwen35_config import QwenConfig


def generate(model, embeddings, tokenizer, prompt_ids, max_new_tokens, max_seq,
             torch, *, compile_decode=False, cuda_graph=False, warmup=0):
    """Generate with an already allocated GPU model; setup is timed separately."""
    if not prompt_ids or min(max_new_tokens, max_seq) <= 0 or max_seq > 8192:
        raise ValueError("A nonempty prompt and positive token/context counts (context <=8192) are required")
    if len(prompt_ids) + max_new_tokens - 1 > max_seq:
        raise ValueError("Prompt and requested generation exceed --max-seq")
    if warmup < 0 or (warmup and not (compile_decode or cuda_graph)):
        raise ValueError("--warmup must be nonnegative and requires --compile or --cuda-graph")
    eos = {tid for name in ("<|im_end|>", "<|endoftext|>")
           if (tid := tokenizer.token_to_id(name)) is not None}

    def embed(ids):
        return np.ascontiguousarray(embeddings[np.asarray(ids, dtype=np.int64)], dtype=np.float32)

    report = {"setup_seconds": 0.0, "decode_seconds": 0.0, "decode_steps": 0}
    with torch.no_grad():
        torch.cuda.synchronize()
        start = time.perf_counter()
        logits = model.run(embed(prompt_ids), past_len=0)
        torch.cuda.synchronize()
        report["prefill_seconds"] = time.perf_counter() - start
        nxt = int(logits[-1].argmax())
        past, generated, printed = len(prompt_ids), [], ""
        use_static = (compile_decode or cuda_graph) and max_new_tokens > 1 and nxt not in eos
        if use_static:
            start = time.perf_counter()
            embed_gpu = torch.from_numpy(np.array(embeddings, copy=True)).to("cuda")
            model.alloc_static(max_seq=max_seq)
            model.prime_static(past)
            model.g_pos.fill_(past)
            model.g_inp.copy_(embed_gpu[nxt].float().reshape(1, model.H))
            if compile_decode:
                # One execution materializes compilation; all setup mutates only static state.
                model.compile_forward(warmup=1)
            for _ in range(warmup):
                model.decode_step(embed_gpu[nxt].float(), past, use_graph=False)
            if cuda_graph:
                model.capture()
            model.prime_static(past)
            torch.cuda.synchronize()
            report["setup_seconds"] = time.perf_counter() - start

        for step in range(max_new_tokens):
            generated.append(nxt)
            text = tokenizer.decode(generated)
            if not text.endswith("\ufffd"):
                print(text[len(printed):], end="", flush=True)
                printed = text
            if nxt in eos or step + 1 == max_new_tokens:
                break
            if past >= max_seq:
                raise ValueError("Decode position exceeds --max-seq")
            torch.cuda.synchronize()
            start = time.perf_counter()
            if use_static:
                logits = model.decode_step(embed_gpu[nxt].float(), past, use_graph=cuda_graph)
                nxt = int(logits[0].argmax().item())
            else:
                logits = model.run(embed([nxt]), past_len=past)
                nxt = int(logits[-1].argmax())
            torch.cuda.synchronize()
            report["decode_seconds"] += time.perf_counter() - start
            report["decode_steps"] += 1
            past += 1
        text = tokenizer.decode(generated)
        print(text[len(printed):], flush=True)
    report.update(token_ids=generated, text=text, generated_tokens=len(generated),
                  decode_tokens_per_second=(report["decode_steps"] / report["decode_seconds"]
                                            if report["decode_seconds"] > 0 else None))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Local prepared decoder ONNX file with adjacent config.json")
    parser.add_argument("--embeddings", type=Path, help="Local FP16 embedding array; defaults to model sibling embeddings.npy")
    parser.add_argument("--tokenizer", type=Path, help="Local tokenizer.json; defaults to model sibling tokenizer.json")
    parser.add_argument("--prompt", default="What is the capital of France? Answer briefly.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-seq", type=int, default=4096, help="Positive context capacity, at most 8192")
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument("--compile", dest="compile_decode", action="store_true",
                        help="Opt in to expensive decode compilation, including one setup execution")
    parser.add_argument("--cuda-graph", action="store_true",
                        help="Opt in to graph capture, including its internal setup warmups")
    parser.add_argument("--warmup", type=int, default=0,
                        help="Extra static setup steps; requires --compile or --cuda-graph (default: 0)")
    args = parser.parse_args(argv)
    if min(args.max_new_tokens, args.max_seq) <= 0 or args.max_seq > 8192:
        parser.error("Token/context counts must be positive and --max-seq must be <=8192")
    if args.warmup < 0 or (args.warmup and not (args.compile_decode or args.cuda_graph)):
        parser.error("--warmup must be nonnegative and requires --compile or --cuda-graph")
    try:
        config = QwenConfig.resolve(args.model)
        supported = QwenConfig.preset("9b")
        if any(getattr(config, f.name) != getattr(supported, f.name)
               for f in fields(QwenConfig) if f.name != "max_position_embeddings"):
            raise ValueError("The GPU decoder supports only the dense 9B configuration; 27B is not supported")
        if args.max_seq > config.max_position_embeddings:
            raise ValueError("--max-seq exceeds the model's configured context")
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(args.tokenizer or args.model.parent / "tokenizer.json"))
        ids = tokenizer.encode(chat_prompt(args.prompt, args.no_thinking)).ids
        if not ids or len(ids) + args.max_new_tokens - 1 > args.max_seq:
            raise ValueError("Prompt must be nonempty and fit requested generation within --max-seq")
        embeddings = np.load(args.embeddings or args.model.parent / "embeddings.npy", mmap_mode="r", allow_pickle=False)
        if embeddings.shape != (config.vocab_size, config.hidden_size) or embeddings.dtype != np.float16:
            raise ValueError("Embeddings must be an FP16 array with shape (vocab_size, hidden_size)")
    except (OSError, ValueError, ImportError) as exc:
        parser.error(str(exc))
    try:
        import torch
    except (ImportError, OSError) as exc:
        parser.error(f"GPU generation requires an available torch installation: {exc}")
    try:
        if not torch.cuda.is_available():
            parser.error("GPU generation requires an available CUDA GPU")
        from breeze import Qwen35GpuModel
        torch.cuda.synchronize()
        start = time.perf_counter()
        model = Qwen35GpuModel(str(args.model))
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - start
        try:
            report = generate(model, embeddings, tokenizer, ids, args.max_new_tokens, args.max_seq,
                              torch, compile_decode=args.compile_decode, cuda_graph=args.cuda_graph,
                              warmup=args.warmup)
        finally:
            del model
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        parser.error(f"GPU generation failed: {exc}")
    report.update(load_seconds=load_seconds, model=str(args.model.resolve()),
                  prompt_tokens=len(ids), max_seq=args.max_seq, thinking=not args.no_thinking,
                  compiled=args.compile_decode, cuda_graph=args.cuda_graph, warmup=args.warmup,
                  timing_note="Load, prefill, static setup and decode are separate; decode excludes printing")
    print(json.dumps({k: v for k, v in report.items() if k not in ("token_ids", "text")}, indent=2))
    return report


if __name__ == "__main__":
    main()
