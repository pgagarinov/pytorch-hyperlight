import argparse
from pathlib import Path
import sys
from jupyterlab_ml_devenv.envtool.utils.conda_env_deployment_tool import (
    update_current_conda_env,
)

MAIN_CMD_NAME = "mlenvtool"


def check_if_yaml_file_to_path(file_name):
    path = Path(file_name)
    assert path.suffix == ".yml"
    return path


def get_prog_name() -> str:
    return f"{MAIN_CMD_NAME} {sys.argv[1]}"


def add_cli_args(argument_parser: argparse.ArgumentParser, arg_group_list):

    assert set(arg_group_list).issubset(["conda_env_yaml_transform"])

    if "conda_env_yaml_transform" in arg_group_list:
        argument_parser.add_argument(
            "conda_env_yaml_transform",
            help="Conda yaml transformation to perform",
            choices=["versions_strip", "versions_eq2ge"],
            type=str,
        )
        argument_parser.add_argument(
            "in_yaml_file",
            help="Input conda environment file in yaml format",
            type=check_if_yaml_file_to_path,
        )
        argument_parser.add_argument(
            "out_yaml_file",
            help="Output conda environment file in yaml format",
            type=check_if_yaml_file_to_path,
        )
        argument_parser.add_argument(
            "--except_package_list",
            nargs="+",
            help="Packages to exclude from transformation",
            required=False,
            default=None,
        )


def run_conda_env_yaml_transform(
    conda_env_transform, in_file, out_file, except_package_list
):

    from jupyterlab_ml_devenv.envtool.utils.conda_env_yaml_tools import (
        conda_yaml_versions_strip,
        conda_yaml_versions_eq2ge,
    )

    if conda_env_transform == "versions_strip":
        conda_yaml_versions_strip(
            in_file, out_file, except_name_list=except_package_list
        )
    elif conda_env_transform == "versions_eq2ge":
        conda_yaml_versions_eq2ge(
            in_file, out_file, except_name_list=except_package_list
        )
    else:
        raise ValueError(
            f"Unexpected value of conda_env_transform: {conda_env_transform}"
        )


def conda_env_run_yaml_transform_cli():
    argument_parser = argparse.ArgumentParser(
        prog=get_prog_name(),
        description="Apply a transformation to Conda YAML environment file",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    add_cli_args(argument_parser, ["conda_env_yaml_transform"])
    args = argument_parser.parse_args(args=sys.argv[2:])

    run_conda_env_yaml_transform(
        args.conda_env_yaml_transform,
        args.in_yaml_file,
        args.out_yaml_file,
        args.except_package_list,
    )


def conda_env_cur_update_cli():

    CURRENT_PATH: Path = Path(__file__).absolute()

    CONDA_REQUIREMENTS_FILE = (
        CURRENT_PATH.parents[2] / "mldevenv_conda_requirements.yml"
    )
    argument_parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog=get_prog_name(),
        formatter_class=argparse.RawTextHelpFormatter,
        description="Update current conda environment by installing additional packages",
    )

    argument_parser.add_argument(
        "mode",
        help="What to update\n"
        "Here the last choice is almost the same as update packages\n"
        "save that necessary packages are filtered from python_requirements.yml file as follows,\n"
        "@@@@@ stands for some mode by which filtration of dependencies is performed,\n"
        "namely a dependency having an inline comment is selected only in the case\n"
        "this inline comment includes the mentioned mode in the comma-separated list of modes,\n"
        "all other dependencies not having inline comments are always included\n",
    )

    argument_parser.add_argument(
        "-d", "--debug", action="store_true", help="update with debug logs"
    )

    # noinspection PyShadowingNames
    args: argparse.Namespace = argument_parser.parse_args(args=sys.argv[2:])

    update_current_conda_env(CONDA_REQUIREMENTS_FILE, args.mode, args.debug)


def cli_strategy():
    argument_parser = argparse.ArgumentParser(
        prog=MAIN_CMD_NAME,
        formatter_class=argparse.RawTextHelpFormatter,
        description="Command line entry point for all RE-think modules.",
    )
    argument_parser.add_argument(
        "strategy",
        help="Choose one of possible strategies. To look a help for each strategy "
        f"run with '-h' argument e.g."
        f"\n{MAIN_CMD_NAME} conda_env_yaml_transform -h"
        f"\n{MAIN_CMD_NAME} conda_env_cur_update -h",
        choices=("conda_env_yaml_transform", "conda_env_cur_update"),
    )
    args = argument_parser.parse_args(args=sys.argv[1:2])
    strategy_name_cli_func_map: dict = {
        "conda_env_yaml_transform": conda_env_run_yaml_transform_cli,
        "conda_env_cur_update": conda_env_cur_update_cli,
    }

    cli_func = strategy_name_cli_func_map[args.strategy]
    cli_func()


if __name__ == "__main__":
    cli_strategy()
