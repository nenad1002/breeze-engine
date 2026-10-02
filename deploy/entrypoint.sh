#!/bin/sh
set -eu

# Compile on the deployment CPU, not on an unrelated image-builder host.
native=true
for arg in "$@"; do
    if [ "$arg" = "--demo" ] || [ "$arg" = "--help" ]; then
        native=false
    fi
done
if [ "$native" = true ]; then
    python -B /app/build_kernel.py
fi
exec python -B /app/serve_breeze.py "$@"