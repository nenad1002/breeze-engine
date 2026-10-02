"""Keep source dependencies, filenames, and user-facing entry points explicit."""
import ast
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
DEPENDENCIES = {"numpy", "onnx", "tokenizers", "torch", "triton", "safetensors", "pytest",
                "fastapi", "starlette", "uvicorn", "pydantic", "httpx2", "playwright"}
TOOLS = (
    "benchmark_cpu_matmul.py", "benchmark_gpu_matmul.py", "benchmark_qwen35_cpu.py",
    "benchmark_qwen35_prefill.py", "build_kernel.py", "generate_qwen35.py",
    "generate_qwen35_gpu.py", "measure_process.py", "prepare_qwen35.py", "profile_graph.py",
    "serve_breeze.py",
)


def python_sources():
    return sorted([*ROOT.glob("*.py"), *(ROOT / "breeze").glob("*.py"),
                   *(ROOT / "tests").glob("*.py")])


def test_dependency_boundary():
    local = {path.stem for path in python_sources()} | {"breeze", "tests"}
    allowed = sys.stdlib_module_names | DEPENDENCIES | local
    for path in python_sources():
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                modules = [node.module]
            else:
                continue
            for module in modules:
                assert module.split(".")[0] in allowed, (path.name, module)


def test_relative_module_paths_exist():
    for path in (ROOT / "breeze").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.ImportFrom) or node.level != 1:
                continue
            modules = [node.module] if node.module else [alias.name for alias in node.names]
            for module in modules:
                assert (path.parent / (module.split(".")[0] + ".py")).is_file(), (path.name, module)


@pytest.mark.parametrize("filename", ["README.md", "SERVICE.md"])
def test_documented_local_links_exist(filename):
    document = ROOT / filename
    for target in re.findall(r"\]\(([^)]+)\)", document.read_text()):
        if "://" not in target:
            assert (document.parent / target.split("#")[0]).exists(), (filename, target)


@pytest.mark.parametrize("filename", TOOLS)
def test_tool_help_needs_no_checkpoint_or_accelerator(filename):
    result = subprocess.run(
        [sys.executable, "-B", str(ROOT / filename), "--help"],
        cwd=ROOT, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "usage:" in result.stdout.lower()


def test_native_build_sources_exist():
    for filename in ("quantized_matmul.cpp", "phi35_decoder.cpp", "qwen35_decoder.cpp"):
        assert (ROOT / "kernel" / filename).is_file()
        assert filename in (ROOT / "build_kernel.py").read_text()


def test_standalone_native_benchmark_uses_current_abi(tmp_path):
    compiler = shutil.which("g++")
    library = ROOT / "breeze" / "_breeze_cpu.so"
    if compiler is None or not library.is_file():
        pytest.skip("Build the native library and install g++ to check the standalone driver")
    executable = tmp_path / "benchmark_matmul"
    subprocess.run(
        [compiler, "-O2", "-std=c++17", str(ROOT / "kernel" / "benchmark_matmul.cpp"),
         str(library), "-Wl,-rpath," + str(library.parent), "-o", str(executable)],
        check=True, capture_output=True, text=True, timeout=60,
    )
    result = subprocess.run([str(executable), "1", "1"], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "threads=1 M=5:" in result.stdout


def test_project_license_and_web_assets_exist():
    for filename in ("LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md"):
        assert (ROOT / filename).is_file()
    for filename in ("index.html", "app.css", "app.js"):
        assert (ROOT / "breeze" / "web" / filename).is_file()


def test_project_author_and_third_party_notices_are_distinct():
    notice = (ROOT / "NOTICE").read_text()
    readme = (ROOT / "README.md").read_text()
    assert "Copyright 2026 Nenad Banfic" in notice
    assert "Created and maintained by **Nenad Banfic**" in readme
    assert "[NOTICE](NOTICE)" in readme
    assert "[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)" in readme


def test_public_performance_summary_reports_measured_relative_throughput():
    text = (ROOT / "README.md").read_text()
    summary = text.split("## Relative generation performance\n", 1)[1].split("\n## ", 1)[0]
    assert "Measured relative generation throughput (Breeze / baseline)" in summary
    assert "1.62x" in summary and "1.01x" in summary
    assert "llama.cpp" in summary
    assert "Qwen" not in summary and "9B" not in summary and "27B" not in summary
    assert len(summary.splitlines()) <= 8
    assert not re.search(r"tokens/s|tok/s|GB/s|GiB", summary)


def test_public_docs_do_not_include_internal_release_material():
    for name in ("PRODUCT.md", "QWEN35_CPU.md", "docs/PUBLIC_RELEASE_AUDIT.md"):
        assert not (ROOT / name).exists()
        for document in ("README.md", "SERVICE.md"):
            assert name not in (ROOT / document).read_text()


def test_source_tree_has_no_external_graph_runtime_references():
    runtime_name = re.compile("onnx" + r"[\s_-]*" + "runtime|onnx" + "rt", re.IGNORECASE)
    short_name = re.compile(r"\b(?:O" + "RT|" + "o" + r"rt)\b|O" + r"rt[A-Z][A-Za-z0-9_]*")
    extensions = {".py", ".md", ".txt", ".tex", ".cpp", ".h", ".js", ".html", ".css",
                  ".yaml", ".yml", ".service"}
    roots = [ROOT, ROOT / "breeze", ROOT / "deploy", ROOT / "kernel", ROOT / "tests"]
    checked = set()
    for source_root in roots:
        paths = source_root.glob("*") if source_root == ROOT else source_root.rglob("*")
        for path in paths:
            if path == ROOT / "README.md":
                continue
            if not path.is_file() or path in checked:
                continue
            if path.suffix not in extensions and path.name not in {"Dockerfile"}:
                continue
            checked.add(path)
            source = path.read_text(encoding="utf-8")
            assert runtime_name.search(source) is None, path
            assert short_name.search(source) is None, path