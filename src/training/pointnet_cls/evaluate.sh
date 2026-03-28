#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source /storage/student6/anaconda3/bin/activate pointnet
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

exec python -u "${SCRIPT_DIR}/evaluate.py" "$@"
