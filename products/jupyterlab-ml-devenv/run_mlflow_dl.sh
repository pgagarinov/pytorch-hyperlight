#!/bin/bash
set -e
source /opt/miniconda/bin/activate
conda activate $1
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR/..
mlflow ui --host 0.0.0.0
