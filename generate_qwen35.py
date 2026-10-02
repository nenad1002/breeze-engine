"""Standalone greedy Qwen3.5-9B/27B generation on the Breeze CPU decoder."""
import argparse
import json
import os
from pathlib import Path
import time

from breeze.chat import chat_prompt
from breeze.qwen35_config import QwenConfig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", help="Local config.json or explicit 9b/27b preset")
    parser.add_argument("--embeddings", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--prompt", default="What is the capital of France? Answer briefly.")
    parser.add_argument("--no-thinking", action="store_true", help="Use the official non-thinking chat suffix")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-seq", type=int, default=4096)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=48)
    parser.add_argument("--int8-activations", action="store_true", help="Use INT8 activations; INT16 is the default")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)
    if min(args.max_new_tokens, args.max_seq, args.chunk_size, args.threads) <= 0:
        parser.error("Token, context, chunk and thread counts must be positive")
    if args.json_out and (args.json_out.exists() or args.json_out.is_symlink()):
        parser.error("--json-out already exists; choose a new result path")
    try:
        config = QwenConfig.resolve(args.model, args.config)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.max_seq > config.max_position_embeddings:
        parser.error("--max-seq exceeds the model's configured context")
    # Set OpenMP before loading the native library.
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ.setdefault("OMP_PROC_BIND", "spread")
    os.environ.setdefault("OMP_PLACES", "cores")
    os.environ.setdefault("OMP_WAIT_POLICY", "active")
    if os.environ.get("I4_QWEN_MAXL") is not None:
        parser.error("Unset I4_QWEN_MAXL; partial-layer debug output is not valid generation")
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(args.tokenizer or args.model.parent / "tokenizer.json"))
    text = chat_prompt(args.prompt, args.no_thinking)
    ids = tok.encode(text).ids
    if not ids:
        parser.error("The tokenized prompt must not be empty")
    if len(ids) + args.max_new_tokens - 1 > args.max_seq:
        parser.error("Prompt and requested generation exceed --max-seq")
    from breeze import Qwen35CpuModel
    from breeze import cpu_backend
    cpu_backend.set_threads(args.threads)
    eos = {token for name in ("<|im_end|>", "<|endoftext|>") if (token := tok.token_to_id(name)) is not None}
    report = {"model": str(args.model.absolute()), "threads": args.threads,
              "prompt": args.prompt, "thinking": not args.no_thinking,
              "activation_bits": 8 if args.int8_activations else 16, "prompt_tokens": len(ids),
              "chunk_size": args.chunk_size, "max_seq": args.max_seq,
              "timing_note": "Load, prefill and decode-forward times are separate; decode excludes tokenization and printing"}
    start = time.perf_counter()
    with Qwen35CpuModel(str(args.model), config=config, embed_path=args.embeddings,
                        max_seq=args.max_seq, hi_prec=not args.int8_activations) as model:
        report["load_seconds"] = time.perf_counter() - start
        report.update(layers=model.L, hidden_size=model.H)
        start = time.perf_counter()
        for offset in range(0, len(ids), args.chunk_size):
            emb = model.embed(ids[offset:offset + args.chunk_size])
            logits = model.run(emb, offset)
        report["prefill_seconds"] = time.perf_counter() - start
        past, generated, decode_times, printed = len(ids), [], [], ""
        for step in range(args.max_new_tokens):
            token = int(logits[-1].argmax())
            generated.append(token)
            decoded = tok.decode(generated)
            if not decoded.endswith("\ufffd"):
                print(decoded[len(printed):], end="", flush=True)
                printed = decoded
            if token in eos or step + 1 == args.max_new_tokens:
                break
            emb = model.embed([token])
            start = time.perf_counter()
            logits = model.run(emb, past)
            decode_times.append(time.perf_counter() - start)
            past += 1
        print(tok.decode(generated)[len(printed):], end="")
        print()
        report.update(generated_tokens=len(generated), token_ids=generated, text=tok.decode(generated),
                      decode_steps=len(decode_times), decode_seconds=sum(decode_times),
                      decode_tokens_per_second=len(decode_times) / sum(decode_times) if sum(decode_times) > 0 else None)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("x") as output:
            output.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("token_ids", "text")}, indent=2))
    return report


if __name__ == "__main__":
    main()