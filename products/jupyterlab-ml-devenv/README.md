PyTorch HyperLight ML Development Environment is a set of modern ML-oriented conda and pip packages with versions carefully chosen to make sure a seamless and conflict-free integration. 

## Prerequisites Features
1. Linux operating system. The testing and development is done in Manjaro Linux. Windows and MacOS are not supported and there are no such plans. This limitation will be less critical once the docker container becomes available.
2. Conda 4.9 or later. The recommended conda distribution is Miniconda. The environment is based on Conda, conda channels (mostly `conda-forge`) are preferred to PyPI where possible. The recommended way of installing Miniconda on Manjaro Linux is [install_miniconda.sh](https://github.com/Alliedium/awesome-linux-config/blob/master/manjaro/basic/install_miniconda.sh) script.

## Features
1. The only IDE provided by the environment is JupyterLab. 
2. We try and usually we are on the bleeding edge in terms of package versions used. The closes analog to PyTorch HyperLight MLDevEnv project is (ml-workspace)[https://github.com/ml-tooling/ml-workspace] project. What makes MLDevEnv different is that packages included into MLDevEnv are usually MUCH more recent comparing to the packages in ml-workspace. This becomes possible because
 - We only support PyTorch and do not support other ML frameworks such as TensorFlow, MXNet etc.
 - We only support JupyterLab
 - We do not try to provide neither ssh, VNC or any other remote desktop tools.

## Installation
Run the following commands without `sudo`:
```bash
source /opt/miniconda/bin/activate
conda env create -n ml-devenv python=3.8
conda activate ml-devenv
./install_all.sh
./init_dl.sh ml-devenv
```
Optionally run `./init_dl.sh ml-devenv` to launch the JupyterLab server, TensorBoard server and MLFlow server.

## MLEnvTool usage
### Examples
- Replace "==" with ">=" for all packages except for pytorch and
torchvision, write the result to 'out.yml'
 `mlenvtool conda_env_yaml_transform versions_eq2ge ./mldevenv_conda_requirements.yml ./out.yml --except_package_list pytorch torchvision`

- Strip versions for all packages except for pytorch and
torchvision, write the result to 'out.yml'
 `mlenvtool conda_env_yaml_transform versions_strip ./mldevenv_conda_requirements.yml ./out.yml --except_package_list pytorch torchvision`  
