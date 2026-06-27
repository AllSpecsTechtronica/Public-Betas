#!/usr/bin/env bash
# Launch Insight/main.py (FastAPI + aiortc WebRTC stack). macOS / Linux.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
INSIGHT_ROOT="${REPO_ROOT}/Insight"
VENV_PY="${REPO_ROOT}/.venv/bin/python"
MAIN_PY="${INSIGHT_ROOT}/main.py"

if [[ ! -x "${VENV_PY}" ]]; then
  echo "Missing venv at ${REPO_ROOT}/.venv. Run scripts/install_packages.sh first." >&2
  exit 1
fi
if [[ ! -f "${MAIN_PY}" ]]; then
  echo "Missing ${MAIN_PY}" >&2
  exit 1
fi

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${INSIGHT_ROOT}"
exec "${VENV_PY}" "${MAIN_PY}" "$@"
