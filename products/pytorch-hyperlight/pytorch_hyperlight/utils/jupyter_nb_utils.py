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

import shutil
import tempfile
from pathlib import Path

import papermill as pm
from tqdm.auto import tqdm

DEFAULT_WORKSPACE_SUBDIR_NAME = "_notebook_workspace"


def get_notebook_list(root_dir):
    notebook_files = (
        set(Path(root_dir).glob("**/[!_]*.ipynb"))
        - set(Path(root_dir).glob("**/.ipynb_checkpoints/**/*.ipynb"))
        - set(Path(root_dir).glob("**/ViT-pytorch/**/*.ipynb"))
    )

    return notebook_files


def run_notebook(notebook_filename, working_dir, fast_dev_run=False, out_file=None):
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        assert len(tmp_dir_name) > 1
        if out_file is None:
            out_file = Path(tmp_dir_name) / notebook_filename.name
        Path.mkdir(working_dir, exist_ok=True)
        pm.execute_notebook(
            notebook_filename, out_file, {"FAST_DEV_RUN": fast_dev_run}, cwd=working_dir
        )


def run_all_notebooks_in_dir(root_dir, notebook_working_dir=None, fast_dev_run=False):
    root_dir = Path(root_dir)
    if notebook_working_dir is None:
        notebook_working_dir = root_dir / WORKSPACE_SUBDIR_NAME
    notebook_list = get_notebook_list(root_dir)
    t_list = tqdm(notebook_list)
    for file_name in t_list:
        msg = f"Running {file_name.name}"
        t_list.write(msg)
        bak_file_name = file_name.with_suffix(".ipynb.bak")
        shutil.copy(file_name, bak_file_name)
        run_notebook(
            file_name,
            notebook_working_dir,
            fast_dev_run=fast_dev_run,
            out_file=file_name,
        )
