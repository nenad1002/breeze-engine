"""Build the CPU shared library; no model downloads or inference."""
import argparse
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "breeze/_breeze_cpu.so")
    parser.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + f".{os.getpid()}.tmp")
    command = [args.cxx, "-O3", "-std=c++17", "-march=native", "-fopenmp", "-shared", "-fPIC",
                             *(str(root / "kernel" / name) for name in
                                 ("quantized_matmul.cpp", "phi35_decoder.cpp", "qwen35_decoder.cpp")),
               "-o", str(temporary)]
    try:
        subprocess.run(command, check=True)
        os.replace(temporary, args.output)  # safe for processes using the previous inode
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Built {args.output} for this host CPU (AVX-512 VNNI/VBMI required).")


if __name__ == "__main__":
    main()