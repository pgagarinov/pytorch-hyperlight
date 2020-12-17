#!/bin/bash
set -e
python ./conda_env_tool.py update all
jupyter serverextension enable --py jupyterlab_code_formatter
pip uninstall -y dataclasses

