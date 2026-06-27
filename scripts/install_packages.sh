#!/usr/bin/env bash
# Create/update the repo virtualenv and install cvLayer Python dependencies (macOS / Linux).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
REQ_CORE="${REPO_ROOT}/requirements-cvlayer.txt"
REQ_RAG="${REPO_ROOT}/requirements-cvlayer-rag.txt"

WITH_RAG=0
PYTHON_CMD="${CVLAYER_PYTHON:-}"

usage() {
  echo "Usage: $(basename "$0") [--with-rag] [--python /path/to/python3]" >&2
  echo "  --with-rag   Also install requirements-cvlayer-rag.txt (Notes / RAG stack)." >&2
  echo "  Environment: CVLAYER_PYTHON overrides the Python used to create the venv." >&2
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-rag)
      WITH_RAG=1
      shift
      ;;
    --python)
      shift
      PYTHON_CMD="${1:-}"
      [[ -n "${PYTHON_CMD}" ]] || usage
      shift
      ;;
    -h|--help) usage ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      ;;
  esac
done

if [[ -z "${PYTHON_CMD}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD="$(command -v python)"
  else
    echo "No python3/python on PATH. Set CVLAYER_PYTHON or pass --python." >&2
    exit 1
  fi
fi

if [[ ! -f "${REQ_CORE}" ]]; then
  echo "Missing ${REQ_CORE}" >&2
  exit 1
fi

echo "[cvLayer] Using Python: ${PYTHON_CMD}"
echo "[cvLayer] Venv: ${VENV_DIR}"

if [[ ! -d "${VENV_DIR}" ]]; then
  "${PYTHON_CMD}" -m venv "${VENV_DIR}"
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "${REQ_CORE}"

if [[ "${WITH_RAG}" -eq 1 ]]; then
  if [[ ! -f "${REQ_RAG}" ]]; then
    echo "Missing ${REQ_RAG}" >&2
    exit 1
  fi
  python -m pip install -r "${REQ_RAG}"
fi

echo "[cvLayer] Done. Activate with: source \"${VENV_DIR}/bin/activate\""
