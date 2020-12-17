#!/bin/bash
set -e
#
screen -wipe || true
killall screen || true
pkill -9 -f mlflow || true
#
rm -f ./screenlog.0
screen -S dl0 -L -d -m ./run_jupyterlab_dl.sh $1
screen -S mlflow0 -L -d -m ./run_mlflow_dl.sh $1
screen -S tb0 -L -d -m ./run_tensorboard_dl.sh $1
sleep 1
screen -ls
sleep 1
sync
watch cat ./screenlog.0
