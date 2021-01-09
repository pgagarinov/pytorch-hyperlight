#!/bin/bash
mlenvtool conda_env_yaml_transform versions_eq2ge ./mldevenv_conda_requirements.yml ./mldevenv_conda_requirements_no_ver.yml --except_package_list 'pytorch' 'torchvision'
