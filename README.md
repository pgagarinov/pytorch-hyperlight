# PyTorch Hyperlight

**The ML micro-framework built on top of [PyTorch-Lightning](https://pytorchlightning.ai/) and [Ray Tune](https://docs.ray.io/en/master/tune/) to push the boundaries of simplicity even further.**


*PyTorch Hyperlight provides* 
  * a growing set of reusable components for running train-validate-test cycles and hyper-parameter tunning for your models with less amount of repeated source code. Just inherit from one of the base classes provided by PyTorch-Hyperlight, inject your models into the superclass constructor and concentrate on improving your model quality, not on the boilerplace code!*
  
  * a set of configuration scripts for setting up a [PyTorch Hyperlight MLDevEnv](https://github.com/pgagarinov/pytorch-hyperlight/blob/main/products/jupyterlab-ml-devenv/README.md) - the JupyterLab-based machine learning environment for Python formed as a set of well-integrated up-to-date ML packages. See [PyTorch Hyperlight MLDevEnv conda environment yaml file](https://github.com/pgagarinov/pytorch-hyperlight/blob/main/products/jupyterlab-ml-devenv/mldevenv_conda_requirements.yml) for the list of packages we support.
 
***Start by cloning of the examples below and modifying it for your needs!***
  
| | | | |
|-|-|-|-|
| [📜 Natural Language Processing](#natural-language-processing)|[🔍 Image Classification](#image-classification) |[🌀 Semantic Segmentation](#semantic-segmentation) | [:loop: Neural Style Transfer](#neural-style-transfer)|
|[<img src="https://user-images.githubusercontent.com/4868370/108255335-2de09900-716d-11eb-8c79-70d32de4c99b.png" width="300">](#natural-language-processing)|[<img src="https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png" width="300">](#image-classification) |[<img src="https://user-images.githubusercontent.com/4868370/108256729-e78c3980-716e-11eb-96f0-789b96f0af4b.png" width="300">](#semantic-segmentation)<br> <sup><sub>source: https://cs.stanford.edu/~roozbeh/pascal-context/</sub></sup> | [<img class="animated-gif" src="https://user-images.githubusercontent.com/4868370/105389270-b6c8eb00-5c28-11eb-9362-dd1c038b18a2.gif" width="300">](#neural-style-transfer)|


## PyTorch Hyperlight key principles
 * *No wheel reinvention*. Parts of PyTorch Lightning or Ray Tune that already provide simple enough APIs are used as is. PyTorch Hyperlight just makes use of those frameworks easier by minimizing an amount of boilerplate code.
 * *Opinionated approach* that doesn't try to be flexible for every possible task. Instead PyTorch Hyperlight tries to address fewer usecase better by providing pre-configured integrations and functionaly out of the box. 
 * *Minimalistic user-facing API* allows to do research by using a single `Runner` class that is similar to PyTorch Lightning's `Trainer` but is a higher-level abstraction.
 * *Expects both model and data as definitions (classes), not as objects*. All this is done to minimize problems with data serialization in Ray Tune which runs trials in separate processes. For
    * *training/validation data* this means that Hyperlight API expects a user to provide *a function that returns DataLoaders*, not ready-to-use DataLoader. Of course you can attach data to your functions with `functools.partion` but this is not recommended;
    * model it means that PyTorch Hyperlight expect the class representing your PyTorch-Lightning module, not the instantiated module object.

## Features 
 * All hyper-parameters are defined in a single dictionary.
 * Plotting of training, validation and testing stage metrics during and after training.
 * Trial comparison reports in form of both tables and graphs.
 * Pre-configured integration with Ray Tune for [ASHA](https://docs.ray.io/en/master/tune/api_docs/schedulers.html#tune-scheduler-hyperband) scheduler and [HyperOpt](https://docs.ray.io/en/latest/tune/api_docs/suggestion.html#hyperopt-tune-suggest-hyperopt-hyperoptsearch) optimization algorithm for out of the box hyper-parameter tuning.
 * Logging the training progress on console (via [tabule](https://github.com/astanin/python-tabulate) library)
 * Pre-configured integration with WandB that works for both single runs and hyper-parameter optimization runs (via Ray Tune)
 * A growing collection of re-usable base classes (tasks) for different ML domains that you can inherit your PyTorch-Lightning modules from.
 * A growing collection of tested Jupyter notebooks demonstrating various PyTorch-Hyperlight usage scenarious.
 
## Assumptions
As most of opinionated frameworks PyTorch Hyperlight makes few assumptions about the way you organize your code:

* You are familiar with [PyTorch-Lightning](https://pytorchlightning.ai/), if not - refer to [PyTorch Lightning awesome documentation](https://pytorch-lightning.readthedocs.io/en/stable/).

* Metrics that you log from your PyTorch Lightning module should have pre-defined prefixes and suffixes:
     * "val", "train" or "test" ("val_f1_step" for example) as a prefix
     * "epoch" or "step" ("train_f1_epoch" for example) as a suffix
     
* DataLoaders should be returned by a function as a dictionary. The function should have "batch_size" as a regular parameter and "n_workers" as a key word parameter. They reason PyTorch Hyperlight doesn't rely on LightningDataModule from PyTorch Lightning is LightningDataModule might contains some data that would have to be serialized in the master process and de-serialized in each Ray Tune worker (workers are responsible for running hyper-parameter search trials).
* WandB API key is placed in `~/.wandb_api_key` file.

## Examples
### Jupyter notebooks
#### Image classification
1. [Boring MNIST](https://github.com/pgagarinov/pytorch-hyperlight/blob/main/products/examples/image_classification/boring_mnist.ipynb)
2. [Boring MNIST model comparison](https://github.com/pgagarinov/pytorch-hyperlight/blob/main/products/examples/image_classification/boring_mnist_model_comparison.ipynb)
3. [Vision Transformer (ViT) for facial image classification based on gender](https://github.com/pgagarinov/pytorch-hyperlight/blob/main/products/examples/image_classification/face_image_classification_gender_vision_transformer.ipynb)
4. [Hybrid Vision Transformer (ViT) for facial image classification based on gender](https://github.com/pgagarinov/pytorch-hyperlight/blob/main/products/examples/image_classification/face_image_classification_gender_hybrid_vision_transformer.ipynb)
5. [Hybrid Vision Transformer (ViT) for CIFAR100 image classification](https://github.com/pgagarinov/pytorch-hyperlight/blob/main/products/examples/image_classification/cifar100_image_classification_hybrid_vision_transformer.ipynb)
#### Semantic segmentation
1. [Semantic segmentation model comparison](https://github.com/pgagarinov/pytorch-hyperlight/blob/main/products/examples/semantic_segmentation/semantic_segmentation_model_comparison.ipynb)

#### Neural style transfer
1. [Plain simple NST](https://github.com/pgagarinov/pytorch-hyperlight/blob/main/products/examples/nst/plain_simple_nst.ipynb)
2. [Multi-style NST](https://github.com/pgagarinov/pytorch-hyperlight/blob/main/products/examples/nst/multi_style_nst.ipynb)

### Natural language processing
1. [BERT finetuning on a subset of SST2](https://github.com/pgagarinov/pytorch-hyperlight/blob/main/products/examples/nlp/bert_sst2_subset_finetuning.ipynb)

## Installation
PyTorch Lightning doesn't have a pip package just yet so please run the following command to install it directly from git

#### Pip
Just run `pip install pytorch_hyperlight`.
