# PyTorch Hyperlight

**The opinionated micro-framework built as a thin [PyTorch-Lightning](https://pytorchlightning.ai/) and [Ray Tune](https://docs.ray.io/en/master/tune/) wrapper built to push the boundaries of simplicity even further.**


 * *Neither the author nor the project do not have any relation to PyTorch-Lightning Team or Ray project.*
 * *PyTorch Hyperlight is not a fork as it does not modify (and there are no such plans) any of PyTorch-Lightning or Ray Tune code and is built on top of the forementioned frameworks.*

## PyTorch Hyperlight key principles
 * *No wheel reinvention* Parts of PyTorch Lightning or Ray Tune that already provide simple enough interfaces are used as is. PyTorch Hyperlight just makes use of those frameworks easier by minimizing an amount of boilerplate code.
 * *Opinionated approach* that doesn't try to be flexible for every possible task. Instead PyTorch Hyperlight tries to address fewer usecase better by providing pre-configured integrations and functionaly out of the box.
 * *Minimalistic user-facing API* allows to do research by using a single `Runner` class that is similar to PyTorch Lightning's `Trainer` but is a higher-level abstraction.
 * *Expect both model and data as definitions, not as data*. All this is done to minimize problems with Ray Tune which runs trails in separate processes. For
    * training/validation data this means that Hyperlight API expects a user to provide *a function that returns DataLoaders*, not ready-to-use DataLoader. Of course you can attach data to your functions with `functools.partion` but this is not recommended.
    * model it means that Hyperlight API (namely `Runner`'s methods) expects a user to provide a class defining a model, not the model itself.

## Features 
 * All hyper-parameters are defined in a single dictionary.
 * Integrated plotting of training, validation and testing stage metrics.
 * Pre-configured integration with Ray Tune for [ASHA](https://docs.ray.io/en/master/tune/api_docs/schedulers.html#tune-scheduler-hyperband) scheduler and [HyperOpt](https://docs.ray.io/en/latest/tune/api_docs/suggestion.html#hyperopt-tune-suggest-hyperopt-hyperoptsearch) optimization algorithm for out of the box hyper-parameter tuning.
 * Logging the training progress on console (via [tabule](https://github.com/astanin/python-tabulate) library)
 * Pre-configured integration with WandB that works for both single runs and hyper-parameter optimization runs (via Ray Tune)
 
## Assumptions
As most of opinionated frameworks PyTorch Hyperlight makes few assumptions about the way you organize your code:

* You are familiar with [PyTorch-Lightning](https://pytorchlightning.ai/), if not - refer to [PyTorch Lightning awesome documentation](https://pytorch-lightning.readthedocs.io/en/stable/).

* Runner

* Metrics that you log from your PyTorch Lightning module should have pre-defined prefixes and suffixes:
     * "val", "train" or "test" ("val_f1_step" for example) as a prefix
     * "epoch" or "step" ("train_f1_epoch" for example) as a suffix
     
* DataLoaders should be returned by a function as a dictionary. The function should have "batch_size" as a regular parameter and "n_workers" as a key word parameter. They reason PyTorch Hyperlight doesn't rely on LightningDataModule from PyTorch Lightning is LightningDataModule might contains some data that would have to be serialized in the master process and de-serialized in each Ray Tune worker (workers are responsible for running hyper-parameter search trials).
* WandB API key should be in the plain text file `~/.wandb_api_key`


## Getting started

#### 1. Define `configure_dataloaders` function that returns your dataloaders as a dictionary:
<details>
  <summary>Source code</summary>
  
  ```python
   import pytorch_lightning as pl
   from torch.utils.data import DataLoader, random_split
   from torchvision.datasets.mnist import MNIST
   from torchvision import transforms
   import pathlib

   EXPERIMENT_ID = "boring-mnist"

   DATASETS_PATH = pathlib.Path(__file__).parent.absolute()

   def create_datasets(val_size=0.2):
       SEED = 16
       pl.seed_everything(SEED)
       #
       full_train_dataset = MNIST(
           DATASETS_PATH, train=True, download=True, transform=transforms.ToTensor()
       )
       test_dataset = MNIST(
           DATASETS_PATH, train=False, download=True, transform=transforms.ToTensor()
       )
       n_full_train_samples = len(full_train_dataset)
       n_val_samples = int(n_full_train_samples * val_size)
       train_dataset, val_dataset = random_split(
           full_train_dataset, [n_full_train_samples - n_val_samples, n_val_samples]
       )

       n_classes = len(MNIST.classes)

       return {
           "full_train_dataset": full_train_dataset,
           "train_dataset": train_dataset,
           "val_dataset": val_dataset,
           "test_dataset": test_dataset,
           "n_classes": n_classes,
       }

   def configure_dataloaders(batch_size, n_workers=4, val_size=0.2):
       #
       SHUFFLE = True
       SAMPLER = None

       result_dict = create_datasets(val_size)

       for prefix in ["train", "full_train"]:
           result_dict[f"{prefix}_loader"] = DataLoader(
               result_dict[f"{prefix}_dataset"],
               batch_size=batch_size,
               shuffle=SHUFFLE,
               sampler=SAMPLER,
               num_workers=n_workers,
               pin_memory=True,
           )

       for prefix in ["val", "test"]:
           result_dict[f"{prefix}_loader"] = DataLoader(
               result_dict[f"{prefix}_dataset"],
               batch_size=batch_size,
               shuffle=False,
               # num_workers=n_workers,
               pin_memory=True,
           )

       return result_dict
  ```
</details>

#### 2. Define your PyTorch-Lightning module and callbacks (if any):
<details>
  <summary>Source code</summary>
  
  ```python
   import pytorch_lightning as pl
   import pytorch_lightning.metrics as metrics
   from pytorch_lightning import Callback

   import torch
   import torch.nn.functional as F

   from transformers import AdamW, get_linear_schedule_with_warmup


   class LitMetricsCalc(torch.nn.Module):
       def __init__(self, prefix, num_classes):
           super(LitMetricsCalc, self).__init__()
           self.acc = metrics.classification.Accuracy()
           self.f1 = metrics.classification.F1(
               num_classes=num_classes, average="macro"
           )
           self.rec = metrics.classification.Recall(
               num_classes=num_classes, average="macro"
           )
           self.prec = metrics.classification.Precision(
               num_classes=num_classes, average="macro"
           )
           self.prefix = prefix

       def step(self, logit, target):
           probs = torch.softmax(logit, dim=1)
           prefix = self.prefix
           self.acc(probs, target)
           self.f1(probs, target)
           self.prec(probs, target)
           self.rec(probs, target)

           return {
               f"{prefix}_acc": self.acc,
               f"{prefix}_f1": self.f1,
               f"{prefix}_prec": self.prec,
               f"{prefix}_rec": self.rec,
           }


   class BoringMNIST(torch.nn.Module):
       def __init__(self, n_classes):
           super().__init__()
           self.l0 = torch.nn.Linear(28 * 28, 256)
           self.l1 = torch.nn.Linear(256, 128)
           self.l2 = torch.nn.Linear(128, n_classes)

       def forward(self, x):
           x = x.view(x.size(0), -1)
           x = torch.relu(self.l0(x))
           x = torch.relu(self.l1(x))
           x = torch.relu(self.l2(x))
           return x


   class LitBoringMNIST(pl.LightningModule):
       def __init__(self, hparams):
           super().__init__()
           self.hparams = hparams

           n_classes = self.hparams.n_classes

           model = BoringMNIST(n_classes)

           self.model = model
           self.train_metric_calc = LitMetricsCalc("train", num_classes=n_classes)
           self.val_metric_calc = LitMetricsCalc("val", num_classes=n_classes)
           self.test_metric_calc = LitMetricsCalc("test", num_classes=n_classes)
           self.freeze()

       def freeze(self):
           for param in self.model.parameters():
               param.requires_grad = False

       def unfreeze(self):
           for param in self.model.parameters():
               param.requires_grad = True

       def unfreeze_tail(self, ind_layer):
           assert ind_layer >= 0
           ind = ind_layer
           while True:
               if ind == 0:
                   for param in self.model.l2.parameters():
                       param.requires_grad = True
               elif ind == 1:
                   for param in self.model.l1.parameters():
                       param.requires_grad = True
               elif ind == 2:
                   for param in self.model.l0.parameters():
                       param.requires_grad = True
               ind -= 1
               if ind < 0:
                   break

       def configure_optimizers(self):
           optimizer = AdamW(
               self.model.parameters(),
               lr=self.hparams.lr,
               betas=(0.9, 0.999),
               eps=1e-8,
           )

           scheduler = get_linear_schedule_with_warmup(
               optimizer, self.hparams.warmup, self.hparams.n_train_steps
           )
           return [optimizer], [
               {"scheduler": scheduler, "interval": "step", "frequency": 1}
           ]

       def forward(self, inputs):
           logits = self.model(inputs)
           return logits

       def forward_batch(self, batch):
           inputs = batch[0]
           return self(inputs)

       def __calc_loss(self, logits, target, log_label):
           loss = F.cross_entropy(logits, target)
           self.log(
               f"{log_label}_loss",
               loss,
               prog_bar=True,
               on_step=True,
               on_epoch=True,
           )
           return loss

       def __stage_step(self, metric_calc, batch, batch_idx, stage):
           logits = self.forward_batch(batch)
           mval_dict = metric_calc.step(logits, batch[1])
           self.log_dict(mval_dict, prog_bar=True, on_step=True, on_epoch=True)
           loss = self.__calc_loss(logits, batch[1], stage)
           return loss

       def training_step(self, batch, batch_idx):
           return self.__stage_step(
               self.train_metric_calc, batch, batch_idx, "train"
           )

       def test_step(self, batch, batch_idx):
           return self.__stage_step(
               self.test_metric_calc, batch, batch_idx, "test"
           )

       def validation_step(self, batch, batch_idx):
           return self.__stage_step(self.val_metric_calc, batch, batch_idx, "val")

   N_CLASSES = 10
   LMODULE_CLASS = LitBoringMNIST
   GPU_PER_TRIAL = 0.3 * torch.cuda.is_available()
   class UnfreezeModelTailCallback(Callback):
       def __init__(self, epoch_vec):
           super().__init__()
           self.epoch_vec = epoch_vec

       def on_epoch_start(self, trainer, pl_module):
           if trainer.current_epoch <= self.epoch_vec[0]:
               pl_module.unfreeze_tail(0)
           elif trainer.current_epoch <= self.epoch_vec[1]:
               pl_module.unfreeze_tail(1)
           else:
               pl_module.unfreeze()

   pl_callbacks = [UnfreezeModelTailCallback(CONFIG["unfreeze_epochs"])]
   ```
</details>

#### 3. Define your model hyper-parameters and extra parameters that control the way training, validation, testing and hyper-parameter tuning is performed:
 
  ```python
   from ray import tune

   CONFIG = {
       "lr": 6.2e-5,  # Initial learning rate
       "warmup": 200,  # For LinearSchedulerWihtWarmup
       "gradient_clip_val": 0,
       "max_epochs": 30,  # the actual number can be less due to early stopping
       "batch_size": 64, 
       "n_classes": N_CLASSES,
       "unfreeze_epochs": [0, 1]  # 2-phase unfreeze, 
       #    unfreeze the tip of the tail at the start of epoch 0,
       #    then unfreeze one more layer at epoch 1,
   }

   EXTRA_CONFIG = {
       "seed": 16,  # just remove this if you do not want determenistic behavior
       "metric_to_optimize": "val_f1_epoch",  # Ray + PTL Trainer
       "ray_metrics_to_show": [
           "val_loss_epoch",
           "val_f1_epoch",
           "val_acc_epoch",
       ],  # for Ray Tune
       "metric_opt_mode": "max",  # Ray + PTL Trainer
       "cpu_per_trial": 3,  # Ray + DataLoaders
       "gpu_per_trial": GPU_PER_TRIAL,  # for Ray Tune
       "n_checkpoints_to_keep": 1,  # for Ray Tune
       "grace_period": 6,  # for Ray Tune
       "epoch_upper_limit": 45,  # for Ray Tune
       "n_samples": 40,  # for Ray Tune
       "ptl_early_stopping_patience": 7,  # for PTL Trainer
       "ptl_early_stopping_grace_period": 7,  # for PTL Trainer
       "ptl_precision": 32,  # or 16, for PTL Trainer
       "train_loader_name": "train_loader",
       "val_loader_name": "val_loader",
       "test_loader_name": "test_loader",
       "batch_size_main": 32,  # batch size for revalidation and test phases 
       #    that run in the main process after all Ray Tune child processes are finished
       "gpus": -1, # -1 - use GPU if available, 0 - use CPU, 1 - use single GPU, 
           # >=2 - use multiple GPUs
   }

   SEARCH_SPACE_CONFIG = {
       "unfreeze_epochs": [0, 1],
       "lr": tune.uniform(1e-5, 1e-4),
       "warmup": tune.choice([200, 500, 600, 1000]),
       "gradient_clip_val": 0,
       "max_epochs": tune.choice([10, 20, 30]),
       "batch_size": tune.choice([16, 32, 64]),
       "n_classes": N_CLASSES,
   }
   ```

#### 4. Create the experiment/trial runner 

  ```python
   from pytorch_hyperlight import Runner

   runner = Runner(
       configure_dataloaders,
       pl_callbacks=pl_callbacks,  # this optional
       experiment_id=EXPERIMENT_ID,  # optional
   )
   ```

#### 5. Run a single trial (combination of training, validation, revalidation and testing stages), plot the trial metrics and access them as Pandas dataframe:

```python
best_result = runner.run_single_trial(LMODULE_CLASS, CONFIG, TUNE_CONFIG)
```

#### 6. Check the results of single trial

##### Access the best model

```python
best_results["lmodule_best"]
```

##### Plot the trial metrics

```python
best_result["metrics"].plot()
```
<img src="products/pytorch-hyperlight/docs/_images/ph_plot.png" width="800px">

##### Access the trial metrics as Pandas dataframe:

```python
best_result["metrics"].df
```
<img src="products/pytorch-hyperlight/docs/_images/ph_df.png" width="800px">

##### the last observed metrics as Pandas series

```python
best_result["metrics"].series_last
```



#### 7. Run a hyper-parameter search by defining Ray Tune search space and calling `run_hyper_opt` method of the runner

```python
SEARCH_SPACE_CONFIG = {
    "unfreeze_epochs": [0, 1],
    "lr": tune.uniform(1e-5, 1e-4),
    "warmup": tune.choice([200, 500, 600, 1000]),
    "gradient_clip_val": 0,
    "max_epochs": tune.choice([10, 20, 30]),
    "batch_size": tune.choice([16, 32, 64]),
    "n_classes": N_CLASSES,
}
best_result = ptl_ray_runner.run_hyper_opt(
    LMODULE_CLASS,
    SEARCH_SPACE_CONFIG,
    TUNE_CONFIG,
)
```

#### 8. Check the results of single trial

##### Access the best model

```python
best_results["lmodule_best"]
```

##### Check the last observed metrics for the best model:

```python
best_result["metrics_last"]
```

##### Access Ray Tune [ExperimentAnalysis](https://docs.ray.io/en/master/tune/api_docs/analysis.html#experimentanalysis-tune-experimentanalysis) object:

```python
best_result["analysis"]
```

# Examples
## Jupyter notebooks
1. [Boring MNIST](https://github.com/pgagarinov/pytorch-hyperlight/blob/main/products/examples/boring_mnist.ipynb)
