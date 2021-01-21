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

from setuptools import setup, find_packages

# noinspection PyProtectedMember
from pip._internal.req import parse_requirements

# flake8: noqa

NAME = "pytorch_hyperlight"

# noinspection PyTypeChecker
install_reqs = parse_requirements("requirements.txt", "req_install_hack")
reqs = [str(ir.requirement) for ir in install_reqs]

setup(
    name=NAME,
    version="0.2.1",
    description="PyTorch Hyperlight is ML micro-framework built as a thin wrapper around PyTorch-Lightning and Ray Tune frameworks to push the boundaries of simplicity even further.",
    long_description="",
    author="Peter Gagarinov",
    author_email="Peter Gagarinov <pgagarinov@gmail.com>",
    url="https://github.com/pgagarinov/pytorch-hyperlight.git",
    zip_safe=True,
    packages=find_packages(),
    install_requires=reqs,
    entry_points={},
    dependency_links=[],
)
