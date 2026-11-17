#!/usr/bin/env bash
# setup-vllm.sh - recreate vllm-env/ from vllm/requirements.lock with uv.
#
# torch and flashinfer are CUDA 13.0 builds (+cu130), which come from their own
# package indexes rather than PyPI.
set -euo pipefail

ROOT="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
ENV="$ROOT/vllm-env"

if [[ -e "$ENV" ]]; then
  echo "$ENV already exists; move it aside first." >&2
  exit 1
fi

uv venv --python 3.13 "$ENV"
uv pip install --python "$ENV/bin/python" \
  --torch-backend cu130 \
  --extra-index-url https://flashinfer.ai/whl/cu130 \
  --index-strategy unsafe-best-match \
  -r "$ROOT/vllm/requirements.lock"

"$ENV/bin/python" -c 'import vllm, torch; print("vllm", vllm.__version__, "| torch", torch.__version__)'
