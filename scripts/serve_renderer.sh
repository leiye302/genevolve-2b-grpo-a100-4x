#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "$0")/.."
exec .venv-render/bin/python -m genevolve_repro.render_service "$@"
