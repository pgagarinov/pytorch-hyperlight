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
pip_list_jupyter_ver_output="$(pip list|grep 'jupyterlab.*3.0.1')"
if [ -z "$pip_list_jupyter_ver_output" ]; then
  pip_list_jupyter_output="$(pip list|grep 'jupyterlab\ ')"
  echo "!!!!>>>>A rogue version of jupyterlab was found and needs to be removed: '$pip_list_jupyter_output' <<<<!!!!"
  echo "Please try to do it via 'pip uninstall jupyterlab'"
  echo "Or delete the following folders manually"
  echo "/home/peter/.local/bin/jlpm
    /home/peter/.local/bin/jupyter-lab
    /home/peter/.local/bin/jupyter-labextension
    /home/peter/.local/bin/jupyter-labhub
    /home/peter/.local/etc/jupyter/jupyter_notebook_config.d/jupyterlab.json
    /home/peter/.local/lib/python3.8/site-packages/jupyterlab-2.2.9.dist-info/*
    /home/peter/.local/lib/python3.8/site-packages/jupyterlab/*
    /home/peter/.local/share/jupyter/"

else
  echo "$pip_list_jupyter_ver_output is installed"
fi
