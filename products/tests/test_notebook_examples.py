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

    @staticmethod
    def run_notebook(file_name):
        notebook_filename = (
            Path(__file__).parents[2] / "products" / "examples" / file_name
        )
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            out_file = Path(tmp_dir_name) / file_name
            pm.execute_notebook(
                notebook_filename,
                out_file,
                {"FAST_DEV_RUN": True},
            )

    @pytest.mark.forked
    def test_boring_mnist_example(self):
        self.run_notebook("boring_mnist.ipynb")

    @pytest.mark.forked
    def test_boring_mnist_model_comparison(self):
        self.run_notebook("boring_mnist_model_comparison.ipynb")
