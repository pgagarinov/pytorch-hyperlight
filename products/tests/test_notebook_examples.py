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


import papermill as pm
from pathlib import Path
import tempfile
import pytest


class TestExampleNotebooks:
    @pytest.mark.parametrize("usage_mode", ["single-run", "hyper-opt"])
    @pytest.mark.forked
    def test_boring_mnist_example(self, usage_mode):
        FILE_NAME = "boring_mnist.ipynb"
        notebook_filename = (
            Path(__file__).parents[2] / "products" / "examples" / FILE_NAME
        )
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            out_file = Path(tmp_dir_name) / FILE_NAME
            pm.execute_notebook(
                notebook_filename,
                out_file,
                {"USAGE_MODE": usage_mode, "FAST_DEV_RUN": True},
            )
