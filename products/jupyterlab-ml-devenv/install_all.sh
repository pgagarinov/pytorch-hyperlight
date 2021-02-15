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
MSGCOLOR=`tput setaf 48`
NOCOLOR=`tput sgr0`

printf "${MSGCOLOR}Installing conda package via conda from conda-forge channel...${NOCOLOR}\n"
# we install conda so that it is not installed later in setup.py
# because if conda is installed via pip it will make conda 
# not functional as a stand-alone command (which is what we need) 
conda install -y -c conda-forge conda
printf "${MSGCOLOR}Installing conda package via conda from conda-forge channel: done${NOCOLOR}\n\n"

printf "${MSGCOLOR}Installing MLDevEnv management tool...${NOCOLOR}\n"
pip install -e .
printf "${MSGCOLOR}Installing MLDevEnv management tool: done${NOCOLOR}\n\n"

printf "${MSGCOLOR}Installing all dependencies for Jupyter ML development environment...${NOCOLOR}\n"
mlenvtool conda_env_cur_update all
printf "${MSGCOLOR}Installing all dependencies for Jupyter ML development environment: done${NOCOLOR}\n\n"


printf "${MSGCOLOR}Installing enabling jupyterlab extensions...${NOCOLOR}\n"
jupyter server extension enable --py jupyterlab_code_formatter
printf "${MSGCOLOR}Installing enabling jupyterlab extensions: done${NOCOLOR}\n\n"

printf "${MSGCOLOR}Removing dataclasses as it is a part of python 3 now...${NOCOLOR}\n"
pip uninstall -y dataclasses
printf "${MSGCOLOR}Removing dataclasses as it is a part of python 3 now: done${NOCOLOR}\n\n"


printf "${MSGCOLOR}Checking if there is rogue JupyterLab installed...${NOCOLOR}\n"
./check_if_rogue_jupyterlab_is_installed.sh
printf "${MSGCOLOR}Checking if there is rogue JupyterLab installed: done${NOCOLOR}\n\n"
printf "${MSGCOLOR}----------SUCCESS------------${NOCOLOR}\n"



