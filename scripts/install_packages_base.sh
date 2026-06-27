#!/usr/bin/env bash
# Create/update .venv-base and install Base_Cv_program (PyQt5) dependencies. macOS / Linux.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv-base"
REQ_BASE="${REPO_ROOT}/requirements-cvlayer-base.txt"
PYTHON_CMD="${CVLAYER_BASE_PYTHON:-}"

usage() {
  echo "Usage: $(basename "$0") [--python /path/to/python3]" >&2
  echo "  Environment: CVLAYER_BASE_PYTHON overrides the Python used to create the venv." >&2
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
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
    echo "No python3/python on PATH. Set CVLAYER_BASE_PYTHON or pass --python." >&2
    exit 1
  fi
fi

if [[ ! -f "${REQ_BASE}" ]]; then
  echo "Missing ${REQ_BASE}" >&2
  exit 1
fi

echo "[cvLayer-base] Using Python: ${PYTHON_CMD}"
echo "[cvLayer-base] Venv: ${VENV_DIR}"

if [[ ! -d "${VENV_DIR}" ]]; then
  "${PYTHON_CMD}" -m venv "${VENV_DIR}"
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "${REQ_BASE}"

echo "[cvLayer-base] Done. Activate with: source \"${VENV_DIR}/bin/activate\""
