FROM python:3.12-slim

LABEL org.opencontainers.image.title="Breeze" \
      org.opencontainers.image.description="Local CPU-backed text assistant service" \
      org.opencontainers.image.licenses="Apache-2.0"

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends g++ \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 breeze
WORKDIR /app
COPY requirements-cpu.txt requirements-serve.txt ./
RUN python -m pip install -r requirements-serve.txt
COPY --chown=breeze:breeze LICENSE NOTICE THIRD_PARTY_NOTICES.md ./
COPY --chown=breeze:breeze breeze ./breeze
COPY --chown=breeze:breeze kernel ./kernel
COPY --chown=breeze:breeze build_kernel.py serve_breeze.py ./
COPY --chown=breeze:breeze deploy/entrypoint.sh ./entrypoint.sh
USER breeze
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=180s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/readyz', timeout=2)"
ENTRYPOINT ["sh", "/app/entrypoint.sh"]
CMD ["--model", "/models/model.onnx", "--host", "0.0.0.0"]