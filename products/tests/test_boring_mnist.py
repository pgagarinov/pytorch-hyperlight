import matplotlib.patches as patches
import numpy as np
import pytorch_hyperlight as pth
import pytorch_lightning as pl
import pytorch_lightning.metrics as metrics
import torch
# noinspection PyPep8Naming
import torch.nn.functional as F
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from pytorch_lightning import Callback
from ray import tune
# noinspection PyProtectedMember
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets.mnist import MNIST
from transformers import AdamW, get_linear_schedule_with_warmup
import warnings
import pathlib
import pytest


class TestBoringMNIST:

    @pytest.fixture
    def boring_mnist(self):
        FAST_DEV_RUN = True
        EXPERIMENT_ID = "boring-mnist"
        DATASETS_PATH = pathlib.Path(__file__).parent.absolute()
        warnings.filterwarnings("ignore")

        # a dedicated function for creating datasets
        # please note how 'full_train_dataset' is created along with train,
        # val and test datasets
        def create_datasets(val_size=0.2):
            SEED = 16
            pl.seed_everything(SEED)
            #
            full_train_dataset = MNIST(
                DATASETS_PATH,
                train=True,
                download=True,
                transform=transforms.ToTensor(),
            )
            test_dataset = MNIST(
                DATASETS_PATH,
                train=False,
                download=True,
                transform=transforms.ToTensor(),
            )
            n_full_train_samples = len(full_train_dataset)
            n_val_samples = int(n_full_train_samples * val_size)
            train_dataset, val_dataset = random_split(
                full_train_dataset,
                [n_full_train_samples - n_val_samples, n_val_samples],
            )

            n_classes = len(MNIST.classes)

            return {
                "full_train_dataset": full_train_dataset,
                "train_dataset": train_dataset,
                "val_dataset": val_dataset,
                "test_dataset": test_dataset,
                "n_classes": n_classes,
            }

        # a dedicated function for creating dataloaders
        # 'full_train_loader' is created along with standard 3 loaders
        # for training, validation and testing datasets
        def configure_dataloaders(batch_size, n_workers=4, val_size=0.2):
            #
            SHUFFLE = True
            SAMPLER = None
            #
            result_dict = create_datasets(val_size)
            #
            for prefix in ["train", "full_train"]:
                result_dict[f"{prefix}_loader"] = DataLoader(
                    result_dict[f"{prefix}_dataset"],
                    batch_size=batch_size,
                    shuffle=SHUFFLE,
                    sampler=SAMPLER,
                    # num_workers=n_workers, # this is commented out to make debugging work in PyCharm
                    # see https://stackoverflow.com/questions/62341906/pytorch-dataloader-doesnt-work-with-remote-interpreter
                    pin_memory=True,
                )
            #
            for prefix in ["val", "test"]:
                result_dict[f"{prefix}_loader"] = DataLoader(
                    result_dict[f"{prefix}_dataset"],
                    batch_size=batch_size,
                    shuffle=False,
                    # num_workers=n_workers,
                    pin_memory=True,
                )

            return result_dict

        BATCH_SIZE = 32

        loaders_dict = configure_dataloaders(BATCH_SIZE)

        class LitMetricsCalc(torch.nn.Module):
            def __init__(self, prefix, num_classes):
                super(LitMetricsCalc, self).__init__()
                self.acc = metrics.classification.Accuracy()
                # compute_on_step=True,
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
                    # f"{prefix}_acc": self.acc(preds, target),
                    f"{prefix}_acc": self.acc,
                    f"{prefix}_f1": self.f1,
                    f"{prefix}_prec": self.prec,
                    f"{prefix}_rec": self.rec,
                }
                """
                return {
                    f"{prefix}_acc": self.acc(probs, target),
                    f"{prefix}_f1": self.f1(probs, target),
                    f"{prefix}_prec": self.prec(probs, target),
                    f"{prefix}_rec": self.rec(probs, target),
                }
                """

        # %%

        class Backbone(torch.nn.Module):
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

        # %%

        class LitBackbone(pl.LightningModule):
            def __init__(self, hparams):
                super().__init__()
                self.hparams = hparams
                #
                n_classes = self.hparams.n_classes
                #
                model = Backbone(n_classes)
                #
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

            # noinspection PyUnusedLocal
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

        # %%

        N_CLASSES = loaders_dict["n_classes"]
        MODEL_CLASS = LitBackbone

        # %%

        lmodule_builder = pth.LitModuleBuilder(MODEL_CLASS)

        # %%

        CONFIG = {
            "lr": 6.2e-5,  # Initial learning rate
            "warmup": 200,  # for StepLR LR scheduler
            "gradient_clip_val": 0,
            "max_epochs": 30,  # the actual number can be less due to early stopping
            "batch_size": 64,  # the maximum for this model and RTX 2070,
            "n_classes": N_CLASSES,
            "unfreeze_epochs": [0, 1],  #
        }

        TUNE_CONFIG = {
            "metric_to_optimize": "val_f1_epoch",  # Ray + PTL Trainer
            "ray_metrics_to_show": [
                "val_loss_epoch",
                "val_f1_epoch",
                "val_acc_epoch",
            ],  # Ray
            "metric_opt_mode": "max",  # Ray + PTL Trainer
            "cpu_per_trial": 3,  # Ray + DataLoaders
            "gpu_per_trial": 0,  # Ray
            "n_checkpoints_to_keep": 1,  # Ray
            "grace_period": 6,  # Ray
            "epoch_upper_limit": 45,  # Ray
            "n_samples": 40,  # Ray
            # "n_samples": 3,
            "ptl_trainer_patience": 7,  # PTL Trainer
            # "ptl_precision": 16,  # PTL Trainer
            "ptl_precision": 32,  # PTL Trainer
            "train_loader_name": "train_loader",
            "val_loader_name": "val_loader",
            "batch_size_main": 32,
            # batch size for validation runs in the main process once all Ray Tune trials are finished
            "gpus": 0,
        }

        if FAST_DEV_RUN:
            CONFIG["max_epochs"] = 2
            TUNE_CONFIG["n_samples"] = 2
            TUNE_CONFIG["gpu_per_trial"] = 0

        SEARCH_SPACE_CONFIG = {
            "unfreeze_epochs": [0, 1],
            "lr": tune.uniform(1e-5, 1e-4),
            "warmup": tune.choice([200, 500, 600, 1000]),
            "gradient_clip_val": 0,
            "max_epochs": tune.choice([10, 20, 30]),
            "batch_size": tune.choice([16, 32, 64]),
            "n_classes": N_CLASSES,
        }
        if FAST_DEV_RUN:
            SEARCH_SPACE_CONFIG["max_epochs"] = 2
            SEARCH_SPACE_CONFIG["batch_size"] = 32

        def set_seed():
            SEED = 16
            pl.seed_everything(SEED)

        # %%

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

        phl_runner = pth.Runner(
            lmodule_builder,
            configure_dataloaders,
            set_seed,
            pl_callbacks=pl_callbacks,
            is_debug=FAST_DEV_RUN,
            experiment_id=EXPERIMENT_ID,
        )

        return {
            "lmodule_builder": lmodule_builder,
            "configure_dataloaders": configure_dataloaders,
            "set_seed": set_seed,
            "pl_callbacks": pl_callbacks,
            "is_debug": FAST_DEV_RUN,
            "experiment_id": EXPERIMENT_ID,
            "phl_runner": phl_runner,
            "config": CONFIG,
            "tune_config": TUNE_CONFIG,
            "search_space_config": SEARCH_SPACE_CONFIG,
            "loaders_dict": loaders_dict,
        }

    def test_touch_run_single_trial(self, boring_mnist):

        phl_runner = boring_mnist["phl_runner"]
        config = boring_mnist["config"]
        tune_config = boring_mnist["tune_config"]
        best_result = phl_runner.run_single_trial(config, tune_config)
        self.touch_check_results(boring_mnist["loaders_dict"], best_result)

    def test_touch_run_hyper_opt(self, boring_mnist):

        phl_runner = boring_mnist["phl_runner"]
        search_space_config = boring_mnist["search_space_config"]
        tune_config = boring_mnist["tune_config"]

        best_result = phl_runner.run_hyper_opt(
            search_space_config,
            tune_config,
        )
        self.touch_check_results(boring_mnist["loaders_dict"], best_result)

    @staticmethod
    def touch_check_results(loaders_dict, best_result):
        DEVICE = torch.device("cpu")

        # noinspection PyUnresolvedReferences
        def imshow(inp, title=None, plt_ax=plt):
            """Imshow for tensors"""
            inp = inp.numpy().transpose((1, 2, 0))
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            inp = std * inp + mean
            inp = np.clip(inp, 0, 1)
            plt_ax.imshow(inp)
            if title is not None:
                plt_ax.set_title(title)
            plt_ax.grid(False)

        def predict_one_sample(model, inputs, device=DEVICE):
            with torch.no_grad():
                inputs = inputs.to(device)
                model.eval()
                logit = model(inputs).cpu()
                probs = torch.nn.functional.softmax(logit, dim=-1).numpy()
            return probs

        # %%

        # noinspection PyShadowingNames,PyTypeChecker
        def show_some_predictions(loaders_dict, lmodule):
            # noinspection PyTypeChecker
            fig, ax = plt.subplots(
                nrows=3, ncols=3, figsize=(12, 12), sharey=True, sharex=True
            )
            for fig_x in ax.flatten():
                random_characters = int(np.random.uniform(0, 1000))
                im_val, label = loaders_dict["val_dataset"][random_characters]
                img_label = " ".join(
                    map(
                        lambda x: x.capitalize(),
                        MNIST.classes[label].split("_"),
                    )
                )

                imshow(im_val.data.cpu(), title=img_label, plt_ax=fig_x)

                fig_x.add_patch(patches.Rectangle((0, 0), 10, 5, color="white"))
                font0 = FontProperties()
                font = font0.copy()
                font.set_family("fantasy")
                prob_pred = predict_one_sample(lmodule, im_val.unsqueeze(0))
                predicted_proba = np.max(prob_pred) * 100
                y_pred = np.argmax(prob_pred)

                predicted_label = MNIST.classes[y_pred]
                predicted_label = (
                        predicted_label[: len(predicted_label) // 2]
                        + "\n"
                        + predicted_label[len(predicted_label) // 2:]
                )
                predicted_text = "{} : {:.0f}%".format(predicted_label, predicted_proba)

                fig_x.text(
                    1,
                    2,
                    predicted_text,
                    horizontalalignment="left",
                    fontproperties=font,
                    verticalalignment="top",
                    fontsize=8,
                    color="black",
                    fontweight="bold",
                )

        show_some_predictions(loaders_dict, best_result["lmodule_best"])
