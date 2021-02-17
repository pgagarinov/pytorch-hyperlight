# Copyright Peter Gagarinov.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash
set -e
#
screen -wipe || true
killall screen || true
pkill -9 -f mlflow || true
#
rm -f ./screenlog.0
screen -S jupyterlab_mldev -L -d -m ./run_jupyterlab_dl.sh $1 $2
screen -S mlflow_mldev -L -d -m ./run_mlflow_dl.sh $1 $2
screen -S tb_mldev -L -d -m ./run_tensorboard_dl.sh $1 $2
sleep 1
screen -ls
sleep 1
sync
watch cat ./screenlog.0
