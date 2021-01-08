#!/bin/bash
mlenvtool conda_env_yaml_transform versions_eq2ge ./mldevenv_conda_requirements.yml ./out.yml --except_package_list 'pytorch' 'torchvision'
