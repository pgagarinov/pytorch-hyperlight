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


EXAMPLES_FOLDER = Path(__file__).parents[2] / "products" / "examples"


def get_notebook_list():
    glob_iter = EXAMPLES_FOLDER.glob("*.ipynb")
    notebook_files = [x for x in glob_iter if x.is_file()]
    notebook_files = [
        file_name for file_name in notebook_files if not file_name.name.startswith("_")
    ]
    notebook_files = sorted(notebook_files)
    return notebook_files


class TestExampleNotebooks:
    @staticmethod
    def run_notebook(notebook_filename, fast_dev_run=False, out_file=None):
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            assert len(tmp_dir_name) > 1
            if out_file is None:
                out_file = Path(tmp_dir_name) / notebook_filename.name
            pm.execute_notebook(
                notebook_filename,
                out_file,
                {"FAST_DEV_RUN": fast_dev_run},
            )

    @pytest.mark.parametrize("file_name", get_notebook_list(), ids=lambda x: x.name)
    @pytest.mark.forked
    def test_run_examples(self, file_name):
        self.run_notebook(file_name, fast_dev_run=True)
