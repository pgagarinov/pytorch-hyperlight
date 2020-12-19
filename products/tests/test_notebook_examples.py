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
            pm.execute_notebook(notebook_filename, out_file, {"USAGE_MODE": usage_mode})
