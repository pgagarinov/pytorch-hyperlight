```python
source /opt/miniconda/bin/activate
conda env create -n ml-devenv python=3.8
conda activate ml-devenv
./install_all.sh
conda deactivate ml-devenv
conda activate ml-devenv
jupyter lab --no-browser --port 8889
```
