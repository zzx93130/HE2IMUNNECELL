#!/bin/zsh
set -euo pipefail
project_dir="${0:A:h}"
cd "$project_dir"
if [[ ! -x .venv/bin/python ]]; then
  python_executable="${HE2CELL_PYTHON:-}"
  if [[ -z "$python_executable" ]]; then
    if command -v python3.12 >/dev/null; then
      python_executable="$(command -v python3.12)"
    elif [[ -x /opt/miniconda3/envs/singlecell/bin/python ]]; then
      python_executable=/opt/miniconda3/envs/singlecell/bin/python
    else
      print -u2 "Install Python 3.12, or set HE2CELL_PYTHON to its executable."
      exit 1
    fi
  fi
  "$python_executable" -m venv .venv
fi
requirements_file=requirements-mac.txt
[[ ! -f requirements-mac.lock.txt ]] || requirements_file=requirements-mac.lock.txt
.venv/bin/python -m pip install -r "$requirements_file"
mkdir -p models data outputs .runtime/matplotlib
./he2cell doctor
