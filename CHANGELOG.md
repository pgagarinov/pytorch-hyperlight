# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).

## [unreleased.Bugfixes] - YYYY-MM-DD

### Added

### Changed

### Deprecated

### Removed

### Fixed


## [0.2.7] - 2021-02-07

### Added
 - new MLDev Python packages as dependencies
    - gitpython==3.1.12
    - wget==3.2
 - unit tests for load_image_as_resized_tensor  

### Fixed
 - ./check_if_rogue_jupyterlab_is_installed.sh doesn't fail if rogue version is found
 - copy_urls_to_files creates Boto3 session even for https urls
 - utils.image_utils.load_image_as_resized_tensor function fails to produce tensors of equal size in certain scenarious. The may lead to nst pipeline failure for certain image sizes (as NSTImageUrlOrPathDataLoader uses load_image_as_resized_tensor)


## [0.2.6] - 2021-02-03

### Added
  - MLDev dependencies
    - visdom==0.1.8.9
    - dominate==2.6.0
    - ml_collections==0.1.0
    - kornia==0.4.1
    - opencv-python==4.5.1.48
    - visdom==0.1.8.9
    - dominate==2.6.0
  - NST model builder in pytorch_hyperlight.tasks.nst module now checks that style_weight_list has the same length as style_image_list

### Fixed
  - plain_simple_nst.ipynb ignores all stypes except for the first one

## [0.2.5] - 2021-02-02

### Added
- The following packaes were added to MLDev:
   - gym==0.18.0
   - pytorch-lightning-bolts==0.3.0
   - timm==0.3.4
   - ray[serve]==1.1.0
   - pipdeptree==2.0.0

- New copy_file_to_s3 and copy_fileobj_to_s3 functions in pytorch_hyperlight.utils.request_utils package


### Changed
- Improved face_image_classification_gender_vit_vs_efficientnet.ipynb by using data augmentation based Albumentations library, 
added class imbalance plotting and displaying sample images, number of trials for each model is reduced down to 1
- Default number of checkpoints kept by PyTorch-Hyperlight in a single-trial model is reduced down to 2
- Many packages in MLDev has been updated:
   - scikit-learn==0.24.1
   - pandas==1.2.1
   - numpy==1.20.0
   - pip==21.0.1
   - ipython==7.20.0
   - matplotlib==3.3.4
   - tqdm==4.55.1
   - captum==0.3.1
   - xeus-python==0.10.2
   - tensorboard==2.4.1
   - papermill==2.3.1
   - scikit-image==0.18.1
   - notebook==6.2.0
   - jupyterlab==3.0.6
   - jupyterlab-lsp==3.3.0
   - jupyterlab_code_formatter==1.4.3
   - ipympl==0.6.3
   - pytorch-lightning==1.1.6
   - boto3==1.16.63
   - psutil==5.8.0
   - transformers==4.2.2
   - wandb==0.10.17
   - pytest==6.2.2
   - coverage==5.4
- The following packages were downgraded
   - pillow==7.2.0 # because gym package requires the older version

### Deprecated


### Removed
- The following packages were temporarily remoted from MLDev
  - datasets (as it requires tqdm<4.50)

### Fixed
- Classifier head was not trained in face_image_classification_gender_vit_vs_efficientnet.ipynb example due to the incorrect classifier 
head paramter names
- Checkpoint file names in 'singe-trial' mode contain confusing prefix and a value of val_loss of the last epoch step. 
This is fixed by reverting back to the checkpoint naming scheme used by PyTorch Lightning by default. 
- check_if_rogue_jupyterlab_is_installed.sh script in MLDev silently crashes when JupyterLab version doesn't match the expected version

## [0.2.4] - 2021-01-29

### Fixed
- "torch" pipy package was referred to as pytorch in requirements.txt which results in failure to install pytorch_hyperlight via pip


## [0.2.3] - 2021-01-29

### Added
- a jupyter notebook example that compares Vision Transformer and EfficientNet for image classification (facial images, gender prediction)
- NSTImageOrPathDataLoader capable of downloading images from 3 different sources transparently: s3, http and local filesystem
- plain_simple_nst.ipynb example capable of downloading both content and style images from either s3 or via http or from local paths and uploading the styled image back to s3/local path. It also works with local paths
- a few auxilary functions that makes it possible with download  images from s3/http/local path and upload them to s3/local path
- image_loading_sync_sizes.ipynb example demonstraing the image loading with automatic cropping and resizing for Neural Style Transfer
- boto3, validators, pytorch, torchvision dependencies for pytorch_hyperlight
- boto3 dependency for jupyter-mldev
- missing license headers to the newly added files
- execute permissions for the shell scripts for building and uploading the pip package
- the proper CHANGELOG.md


### Changed
- multy_style_nst.ipynb is made even more compact by using the higher-level NST DataLoader capable or downloading images directly from both http and s3 urls


### Fixed
- Fixed: style images are not resized correctly when they are larger than content images


## [0.2.2] - 2021-01-28

### PyTorch-Hyperlight
#### Added
 - a new higher-level image file data loader class for NST called "NSTImageFileDataLoader" is placed into tasks.nst.py package. 
 - new utility functions for loading and showing image tensors in utils.image_utils package and download_urls function in utils.request_utils.py
 - requests and pillow are now dependencies for both pytorch_hyperlight
 - n_steps_per_epoch parameter in hparams calculated by Runner class and provided to PyTorch Lightning modules automatically
 
#### Changed:
 - multi_style_nst.ipynb is made much more lightweight as it now uses the classes from nst.py package.
 - pytorch-lightning minimal version is set to 1.1.5 in requirements.txt (as in this version introduced the important fixes to the progress)
 
 
### JupyterLab-based MLDev environment
#### Added
 - new dependencies: requests, pillow, kaggle cli and pytorch-pretrained-vit
 


## [0.2.1] - 2021-01-21

### PyTorch-Hyperlight
#### Added
 - Progress bar displayed by Runner now features an integrated metrics plotting which makes it unnecessary to call `show_report()` by hand once the training/validation loop is finished
 - Added a multi-style neural style transfer example
 - Added a notebook that runs all the examples. Useful for automating re-running the examples after the major changes.
 
#### Changed:
 - Text logs with current metrics values are only displayed when PyTorch-Hyperlight is used from console. Within the JupyterLab the metrics log messages are replaced with the dynamic plotting integrated with the progress bar
 - The happy path test for the example notebooks now scans the example folder and runs all the notebooks it finds (except for those with name  that starts with "_" ).
 
#### Fixed
 - Training and validation metrics were displayed even on test and revalidation stages inside the dataframes, sometimes this breaks the plotting as well.
 
### JupyterLab-based MLDev environment
#### Fixed
 - Some of the scripts contain hard-coded paths
 - The post-installation checks are not robust enough



## [0.2.0] - 2021-01-12

### PyTorch-Hyperlight
#### Added
 - `Runner` class accumulates all time-series metrics from all subsequent calls of `run_single_trial` and `run_hyperopt`. The metrics be accessed at any point via `get_metrics` method or displayed via `show_metric_report` method. [MNIST model comparison noteboook](https://github.com/pgagarinov/pytorch-hyperlight/blob/main/products/examples/boring_mnist_model_comparison.ipynb) provides the usage examples. The old behavior without any metrics collection is available is still available via `BaseRunner` class. The names of runs (during metrics accumulation) are either generated automatically (based on class names of PyTorch-Lightning modules) or can be specified either explicitly or partially (via suffixes). See the notebook example above for details.
 
 - `AClassificationTask` class is designed in a flexible and re-usable way to serve as a base for PyTorch lightning modules for classification problems. By default `AClassificationTask` calculates a set of classification metrics (f1, accuracy, precision and recall) but can be extended by inheriting from if needed. The only abstract method that is not defined in `AClassificationTask` is `configure_optimizers`. 
 - More specific `ClassificationTaskAdamStepLR` (Adap optimizer + StepLR scheduler) and `ClassificationTaskAdamWWarmup` (AdamW optimizer + LinearWarmup scheduler from [transformers](https://github.com/huggingface/transformers) library) tasks.
 - Dependency on [transformers](https://github.com/huggingface/transformers) library.
 - Tests for the new notebook examples.
 - Tests for the grace period and the subsequent runs for the same PyTorch-Lightning modules.
 - 'stage-list' field in metrics dataframes (TrialMetrics.df for instance).
 - integrated self-checks in `run_hyperopt` and `run_single_trial` that make sure that the metrics returned by PyTorch-Lightning Trainer's methods `fit` and `test` are the same as those accumulated via callbacks by PyTorch-Hyperlight.
 
#### Changed:
 - run_hyperopt method now returns TrialMetrics similarly to run_single_trial method.
 - TrialMetrics class is refactored to accept only a single Pandas DataFrame with time series metrics in the constructor.
 - Dependency on PyTorch-Lightning version is relaxed (`1.1.*` now)
 - Jupyter notebook examples are better documented and cleaner now.
 
#### Fixed
 - 'grace_period' parameter in CONFIG is not recognized by `run_single_trial` method
 - Paths to MNIST dataset are hardcoded in all Jupyter notebook examples
 
 
### JupyterLab-based MLDev environment
 - Switch to PyTorch 1.7.1 and CUDA 11.
 - Switch to JupyterLab 3.0.1.
 - The new `envtool` CLI tool for managing the environment. Usage examples:
    ```python
    # strip conda package versions for all packages but pytorch and torchvision, use /mldevenv_conda_requirements.yml as a source 
    # and put the # resulting file to /out.yml
    mlenvtool conda_env_yaml_transform versions_strip ./mldevenv_conda_requirements.yml ./out.yml --except_package_list 'pytorch' 'torchvision'
    
    # replace `==` with `>=` for all packages except for `pytorch` and `torchvision`
    mlenvtool conda_env_yaml_transform versions_eq2ge ./mldevenv_conda_requirements.yml ./out.yml --except_package_list 'pytorch' 'torchvision'    
    
    # update all conda packages and JupyterLab extensions in the current conda environment
    mlenvtool conda_env_update all
    
    # see help
    mlenvtool -h
    
    # see help for `conda_env_update` command
    mlenvtool conda_env_update -h
    ```
 - `./install_all.sh` script now uses `mlenvtool` and prints cleaner status messages.
 - Almost all Python dependencies have been updated.


## [0.1.3] - 2020-12-24


- "ptl_trainer_grace_period parameter" has been merged with "grace_period" parameter which is now responsible for both Ray Tune Scheduler's early stopping patience and PyTorch Lightning Trainer's early stopping patience
- PyTorch Lightning version has been bumped up to 1.1.2

## [0.1.2] - 2020-12-21

- TrialMetrics.plot method now uses different styles for train and not train stage metrics. Different groups of metrics graphs use different markers.


## [0.1.1] - 2020-12-20
### Fixed
- Ray Tune is missing in requirements.txt
- minor cosmetic issued in comments and README.md


## [0.1.0] - 2020-12-20
This is the very first release

### PyTorch Hyperlight key principles
 * *No wheel reinvention* Parts of PyTorch Lightning or Ray Tune that already provide simple enough interfaces are used as is. PyTorch Hyperlight just makes use of those frameworks easier by minimizing an amount of boilerplate code.
 * *Opinionated approach* that doesn't try to be flexible for every possible task. Instead PyTorch Hyperlight tries to address fewer usecase better by providing pre-configured integrations and functionaly out of the box.
 * *Minimalistic user-facing API* allows to do research by using a single `Runner` class that is similar to PyTorch Lightning's `Trainer` but is a higher-level abstraction.
 * *Expect both model and data as definitions, not as data*. All this is done to minimize problems with Ray Tune which runs trails in separate processes. For
    * training/validation data this means that Hyperlight API expects a user to provide *a function that returns DataLoaders*, not ready-to-use DataLoader. Of course you can attach data to your functions with `functools.partion` but this is not recommended.
    * model it means that Hyperlight API (namely `Runner`'s methods) expects a user to provide a class defining a model, not the model itself.

### Features 
 * All hyper-parameters are defined in a single dictionary.
 * Integrated plotting of training, validation and testing stage metrics.
 * Pre-configured integration with Ray Tune for [ASHA](https://docs.ray.io/en/master/tune/api_docs/schedulers.html#tune-scheduler-hyperband) scheduler and [HyperOpt](https://docs.ray.io/en/latest/tune/api_docs/suggestion.html#hyperopt-tune-suggest-hyperopt-hyperoptsearch) optimization algorithm for out of the box hyper-parameter tuning.
 * Logging the training progress on console (via [tabule](https://github.com/astanin/python-tabulate) library)
 * Pre-configured integration with WandB that works for both single runs and hyper-parameter optimization runs (via Ray Tune)
 
### Assumptions
As most of opinionated frameworks PyTorch Hyperlight makes few assumptions about the way you organize your code:

* You are familiar with [PyTorch-Lightning](https://pytorchlightning.ai/), if not - refer to [PyTorch Lightning awesome documentation](https://pytorch-lightning.readthedocs.io/en/stable/).

* Runner

* Metrics that you log from your PyTorch Lightning module should have pre-defined prefixes and suffixes:
     * "val", "train" or "test" ("val_f1_step" for example) as a prefix
     * "epoch" or "step" ("train_f1_epoch" for example) as a suffix
     
* DataLoaders should be returned by a function as a dictionary. The function should have "batch_size" as a regular parameter and "n_workers" as a key word parameter. They reason PyTorch Hyperlight doesn't rely on LightningDataModule from PyTorch Lightning is LightningDataModule might contains some data that would have to be serialized in the master process and de-serialized in each Ray Tune worker (workers are responsible for running hyper-parameter search trials).
* WandB API key should be in the plain text file `~/.wandb_api_key`
