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

import subprocess
from pathlib import Path
import os
import platform
import sys
import logging
import tempfile
from typing import List, Dict, Union, Any

#
LOG_LEVEL: str = "INFO"
MAX_LINE_LENGTH: int = 88
LOGGING_FORMAT: str = (
    "\n"
    + "-" * MAX_LINE_LENGTH
    + "\n%(asctime)s - %(name)s - %(levelname)s: %(message)s\n"
    + "-" * MAX_LINE_LENGTH
    + "\n"
)
logging.basicConfig(format=LOGGING_FORMAT, level=LOG_LEVEL)

IS_WINDOWS: bool = platform.system() == "Windows"

logger = logging.getLogger(__file__)


# noinspection DuplicatedCode
def exec_cmd(command_list: list) -> int:
    if not IS_WINDOWS:
        command: str = " ".join(command_list)
    else:
        command: list = command_list
    cmd_exec_str: str = f'executing command "{command}"'
    logger.info(cmd_exec_str + "...")
    exec_code = subprocess.call(command, shell=True)
    if exec_code == 0:
        logger.info(cmd_exec_str + ": SUCCESS")
    else:
        logger.fatal(f"command failed, exiting with code {exec_code}")
    return exec_code


# noinspection DuplicatedCode
def exec_cmd_or_exit(command_list: list) -> None:
    try:
        exec_code = exec_cmd(command_list)
    except KeyboardInterrupt:
        logger.fatal("Execution has been interrupted")
        sys.exit()
    except Exception as e:
        logger.fatal(f"Command has failed with the following exception: {e}")
        raise e

    if exec_code != 0:
        sys.exit(exec_code)


def get_current_conda_env_path():
    return os.environ["CONDA_PREFIX"]


def get_current_conda_env_name():
    return os.environ["CONDA_DEFAULT_ENV"]


CURRENT_PATH: Path = Path(__file__).absolute().parent

MAIN_CMD_NAME: str = __file__


def get_prog_name() -> str:
    return MAIN_CMD_NAME + " " + sys.argv[1]


def update_conda_env_from_relfile(
    conda_env_path: str, req_abspath: Union[Path, str], debug: bool = False
) -> None:
    conda_command_envupdate_list: List[str] = [
        "conda",
        "env",
        "update",
        "--debug" if debug else "",
        "-p",
        conda_env_path,
        "-f",
    ] + [str(req_abspath)]

    exec_cmd_or_exit(conda_command_envupdate_list)


def pip_install_modules_by_relpath(module_relpath_list: List[Path]) -> None:
    __PIP_INSTALL_COMMAND_LIST = [
        "pip",
        "install",
        "--trusted-host",
        "pypi.org",
        "--trusted-host",
        "pypi.python.org",
        "--trusted-host",
        "files.pythonhosted.org",
        "-e",
    ]

    for module_path in module_relpath_list:
        exec_cmd_or_exit(__PIP_INSTALL_COMMAND_LIST + [str(CURRENT_PATH / module_path)])


def jlab_install_extensions(extension_name_list) -> None:

    if len(extension_name_list) > 0:
        __JUPYTERLAB_EXTENSION_INSTALL_COMMAND = ["jupyter", "labextension", "install"]
        os.environ["NODE_OPTIONS"] = "--max-old-space-size=4096"

        for extension_name in extension_name_list:
            exec_cmd_or_exit(
                __JUPYTERLAB_EXTENSION_INSTALL_COMMAND + [extension_name, "--no-build"]
            )

        exec_cmd_or_exit(["jupyter", "lab", "build", "--minimize=False"])

        del os.environ["NODE_OPTIONS"]


def filter_python_requirements_by_mode(
    source_conda_env_yaml_file: Path, mode: str
) -> Path:
    out_stream: tempfile.NamedTemporaryFile = tempfile.NamedTemporaryFile(
        "wt", suffix=".yml", delete=False
    )
    with open(str(source_conda_env_yaml_file), "rt") as inp_stream:
        inp_line_str: str = inp_stream.readline()
        if mode == "all":
            while inp_line_str:
                out_stream.write(inp_line_str)
                inp_line_str = inp_stream.readline()
        else:
            is_pip: bool = False
            inp_pip_line_str = None
            prev_out_line_str = None
            while inp_line_str:
                if not is_pip:
                    is_pip = "- pip:" in inp_line_str
                    if is_pip:
                        inp_pip_line_str = inp_line_str.rstrip()
                        inp_line_str = inp_stream.readline()
                        if not inp_line_str:
                            out_stream.write(prev_out_line_str)
                        continue
                ind_comment_start: int = inp_line_str.find("#")
                if ind_comment_start >= 0:
                    mode_list: List[str] = [
                        mode_str.strip()
                        for mode_str in inp_line_str[ind_comment_start + 1 :]
                        .rstrip()
                        .split(",")
                    ]
                    if mode not in mode_list:
                        inp_line_str = inp_stream.readline()
                        if not inp_line_str:
                            out_stream.write(prev_out_line_str)
                        continue
                    out_line_str = inp_line_str[:ind_comment_start]
                else:
                    out_line_str = inp_line_str.rstrip()
                inp_line_str = inp_stream.readline()
                if prev_out_line_str is not None:
                    out_stream.write(prev_out_line_str + "\n")
                if is_pip:
                    out_stream.write(inp_pip_line_str + "\n")
                    is_pip = False
                if inp_line_str:
                    prev_out_line_str = out_line_str
                else:
                    out_stream.write(out_line_str)
    out_stream.close()
    return Path(out_stream.name)


def update_conda_env(conda_env_path, source_conda_env_yaml_file, mode, debug=False):
    __MODE2CMD_DICT: Dict[str, Dict[str, Any]] = {
        "all": {
            "install_requirements": True,
            "filter_requirements": False,
            "install_modules": True,
            "module_names": [],  # empty list means to take all available modules
            "jlab_install_extensions": True,
        },
        "allbutjupyterext": {
            "install_requirements": True,
            "filter_requirements": False,
            "install_modules": True,
            "module_names": [],  # empty list means to take all available modules
            "jlab_install_extensions": False,
        },
        "products": {
            "install_requirements": False,
            "filter_requirements": False,
            "install_modules": True,
            "module_names": [],  # empty list means to take all available modules
            "jlab_install_extensions": False,
        },
        "packages": {
            "install_requirements": True,
            "filter_requirements": False,
            "install_modules": False,
            "module_names": [],
            "jlab_install_extensions": False,
        },
        "jupyterext": {
            "install_requirements": False,
            "filter_requirements": False,
            "install_modules": False,
            "module_names": [],
            "jlab_install_extensions": True,
        },
    }
    if mode in __MODE2CMD_DICT:
        cmd_run_param_dict: Dict[str, Any] = __MODE2CMD_DICT[mode]
    else:
        cmd_run_param_dict: Dict[str, Any] = {
            "install_requirements": True,
            "filter_requirements": True,
            "install_modules": False,
            "module_names": [],
            "jlab_install_extensions": False,
        }

    if cmd_run_param_dict["install_requirements"]:
        python_requirements_abs_path: Path = filter_python_requirements_by_mode(
            source_conda_env_yaml_file,
            mode if cmd_run_param_dict["filter_requirements"] else "all",
        )
        update_conda_env_from_relfile(
            conda_env_path, python_requirements_abs_path, debug
        )
        os.unlink(str(python_requirements_abs_path))

        exec_cmd(["python", "-m", "spacy", "download", "en_core_web_sm"])
        exec_cmd(["python", "-m", "spacy", "download", "en"])
        exec_cmd(["python", "-m", "spacy", "download", "de"])
        exec_cmd(["python", "-m", "spacy", "download", "xx_ent_wiki_sm"])

    if cmd_run_param_dict["install_modules"]:
        module_to_path_dict: Dict[str, Path] = {
            # "cufflinks": Path("externals"),
        }
        module_name_list: List[str] = cmd_run_param_dict["module_names"]
        if len(module_name_list) == 0:
            module_relpath_list = [
                module_relpath / module_name
                for module_name, module_relpath in module_to_path_dict.items()
            ]
        else:
            module_relpath_list = [
                module_relpath / module_name
                for module_name, module_relpath in module_to_path_dict.items()
                if module_name in module_name_list
            ]
        pip_install_modules_by_relpath(module_relpath_list)

    if cmd_run_param_dict["jlab_install_extensions"]:
        jlab_install_extensions([])


def update_current_conda_env(*args, **kwargs):
    current_conda_env_path = get_current_conda_env_path()
    update_conda_env(current_conda_env_path, *args, **kwargs)
