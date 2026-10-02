"""Serve a local CPU checkpoint, or preview the workspace without loading a model."""
import argparse
import json
import os
from pathlib import Path

from breeze.service_config import ServiceSettings, preflight


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", type=Path, help="Prepared local decoder; adjacent config/tokenizer/embeddings required")
    source.add_argument("--demo", action="store_true", help="Clearly labeled, scripted UI preview; no model loaded")
    parser.add_argument("--model-id", default="breeze-local", help="Public API model ID, not a filesystem path")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address; non-loopback requires BREEZE_API_KEY")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-seq", type=int, default=4096)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--max-pending", type=int, default=4, help="Maximum admitted requests, including the active request")
    parser.add_argument("--request-timeout", type=float, default=120.0, help="Soft deadline including queue time")
    parser.add_argument("--allowed-host", action="append", dest="allowed_hosts", help="Exact HTTP hostname or IP; repeat for aliases")
    parser.add_argument("--check", action="store_true", help="Check prerequisites without starting a server or loading weights")
    args = parser.parse_args(argv)
    check = args.check
    values = vars(args).copy()
    values.pop("check")
    if values["allowed_hosts"] is None:
        values.pop("allowed_hosts")
    else:
        values["allowed_hosts"] = tuple(values["allowed_hosts"])
    try:
        settings = ServiceSettings(**values, api_key=os.environ.get("BREEZE_API_KEY"))
    except ValueError as error:
        parser.error(str(error))
    checks = preflight(settings)
    if check:
        print(json.dumps({"checks": checks, "authentication_required": settings.api_key is not None,
                          "mode": "demo" if settings.demo else "native"}, indent=2))
        return 0 if all(item["ok"] for item in checks) else 1
    failures = [item["check"] for item in checks if not item["ok"]]
    if failures:
        parser.error("Preflight failed: " + ", ".join(failures) + "; use --check for details")
    try:
        import uvicorn
        from breeze.server import create_app
    except ImportError:
        parser.error("Serving dependencies are missing; install requirements-serve.txt")
    print("Breeze: scripted demo; no model inference." if settings.demo else
          f"Breeze: CPU inference, model={settings.model_id}, threads={settings.threads}.", flush=True)
    print("One inference worker; bounded queue; no prompt or response logging.", flush=True)
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, workers=1,
                access_log=False, server_header=False, proxy_headers=True,
                forwarded_allow_ips="127.0.0.1,::1", timeout_keep_alive=5,
                limit_concurrency=128, timeout_graceful_shutdown=int(settings.request_timeout) + 30)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())