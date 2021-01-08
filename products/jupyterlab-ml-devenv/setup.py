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

from setuptools import setup, find_namespace_packages
from pip._internal.req import parse_requirements

# flake8: noqa

NAMESPACE = "jupyterlab_ml_devenv"
SERVICE = "envtool"

NAME = "{}.{}".format(NAMESPACE, SERVICE)

install_reqs = parse_requirements("envtool_requirements.txt", "req_install_hack")
reqs = [str(ir.requirement) for ir in install_reqs]

setup(
    name=f"{NAMESPACE}.{SERVICE}",
    version="0.0.1dev",
    description="JupyterLab-based ML development environment",
    long_description="",
    author="Peter Gagarinov",
    author_email="Peter Gagarinov <pgagarinov@gmail.com>",
    zip_safe=True,
    packages=find_namespace_packages(include=[NAMESPACE + ".*"]),
    install_requires=reqs,
    entry_points={"console_scripts": [f"mlenvtool={NAME}.cli:cli_strategy"]},
    dependency_links=[],
)
