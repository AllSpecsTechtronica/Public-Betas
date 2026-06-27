#!/usr/bin/env bash
# Launch Base_Cv_program/main.py (PyQt5 modular CV). macOS / Linux. Uses .venv-base only.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASE_ROOT="${REPO_ROOT}/Base_Cv_program"
VENV_PY="${REPO_ROOT}/.venv-base/bin/python"
MAIN_PY="${BASE_ROOT}/main.py"

if [[ ! -x "${VENV_PY}" ]]; then
  echo "Missing venv at ${REPO_ROOT}/.venv-base. Run scripts/install_packages_base.sh first." >&2
  exit 1
fi
if [[ ! -f "${MAIN_PY}" ]]; then
  echo "Missing ${MAIN_PY}" >&2
  exit 1
fi

cd "${BASE_ROOT}"
exec "${VENV_PY}" "${MAIN_PY}" "$@"
