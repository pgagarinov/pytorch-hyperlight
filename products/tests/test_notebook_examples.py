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


from pathlib import Path
import pytest
from pytorch_hyperlight.utils.jupyter_nb_utils import run_notebook, get_notebook_list

EXAMPLES_FOLDER = Path(__file__).parents[2] / "products" / "examples"
NOTEBOOK_WORKING_DIR = EXAMPLES_FOLDER / "_notebook_workspace"


class TestExampleNotebooks:
    @pytest.mark.parametrize(
        "file_name",
        get_notebook_list(EXAMPLES_FOLDER),
        ids=lambda p: str(p.relative_to(p.parent.parent).with_suffix(''))
    )
    @pytest.mark.forked
    def test_run_examples(self, file_name):
        run_notebook(file_name, NOTEBOOK_WORKING_DIR, fast_dev_run=True)
