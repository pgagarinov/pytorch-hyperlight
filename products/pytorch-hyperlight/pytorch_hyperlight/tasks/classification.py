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

import pytorch_lightning as pl
from pytorch_lightning import metrics
import torch
from abc import abstractmethod
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.nn as nn


class LitMetricsCalc(torch.nn.Module):
    def __init__(self, prefix, num_classes, metric_list=None):
        super(LitMetricsCalc, self).__init__()
        if metric_list is None:
            metric_list = ["acc", "f1"]

        metrics_dict = nn.ModuleDict()
        if "acc" in metric_list:
            metrics_dict["acc"] = metrics.classification.Accuracy()
        if "f1" in metric_list:
            metrics_dict["f1"] = metrics.classification.F1(
                num_classes=num_classes, average="macro"
            )
        if "prec" in metric_list:
            metrics_dict["prec"] = metrics.classification.Precision(
                num_classes=num_classes, average="macro"
            )
        if "rec" in metric_list:
            metrics_dict["rec"] = metrics.classification.Precision(
                num_classes=num_classes, average="macro"
            )
        self._metrics_dict = metrics_dict
        self.prefix = prefix

    def step(self, logit, target):
        if logit.ndimension() == 1:
            probs = torch.sigmoid(logit)
        else:
            probs = torch.softmax(logit, dim=1)

        prefix = self.prefix
        for metric in self._metrics_dict.values():
            metric(probs, target)

        return {
            f"{prefix}_{metric_name}": metric
            for metric_name, metric in self._metrics_dict.items()
        }


class AClassificationTask(pl.LightningModule):
    def __init__(self, hparams, model, criterion):
        super().__init__()
        self.save_hyperparameters(hparams)

        n_classes = self.hparams.n_classes
        if "metric_list" in self.hparams:
            kwargs = {"metric_list": self.hparams["metric_list"]}
        else:
            kwargs = {}

        self.model = model
        self.criterion = criterion
        self.train_metric_calc = LitMetricsCalc(
            "train", num_classes=n_classes, **kwargs
        )
        self.val_metric_calc = LitMetricsCalc("val", num_classes=n_classes, **kwargs)
        self.test_metric_calc = LitMetricsCalc("test", num_classes=n_classes, **kwargs)

    @abstractmethod
    def configure_optimizers(self):
        pass

    def forward(self, inputs):

        logits = self.model(inputs)

        return logits

    def forward_batch(self, batch):
        inputs = batch[0]
        return self(inputs)

    def __calc_loss(self, logits, target, log_label):
        loss = self.criterion(logits, target)
        self.log(f"{log_label}_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    # noinspection PyUnusedLocal
    def __stage_step(self, metric_calc, batch, batch_idx, stage):
        logits = self.forward_batch(batch)
        mval_dict = metric_calc.step(logits, batch[1])
        self.log_dict(mval_dict, prog_bar=True, on_step=True, on_epoch=True)
        loss = self.__calc_loss(logits, batch[1], stage)
        return loss

    def training_step(self, batch, batch_idx):
        return self.__stage_step(self.train_metric_calc, batch, batch_idx, "train")

    def test_step(self, batch, batch_idx):
        return self.__stage_step(self.test_metric_calc, batch, batch_idx, "test")

    def validation_step(self, batch, batch_idx):
        return self.__stage_step(self.val_metric_calc, batch, batch_idx, "val")


class ClassificationTaskAdamStepLR(AClassificationTask):
    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)

        lr_step_epochs = self.hparams.lr_step_epochs
        lr_step_factor = self.hparams.lr_step_factor

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, lr_step_epochs, lr_step_factor
        )
        return [optimizer], [
            {"scheduler": scheduler, "interval": "epoch", "frequency": 1}
        ]


class ClassificationTaskAdamWWarmup(AClassificationTask):
    def configure_optimizers(self):

        warmup = self.hparams.warmup
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer, warmup, self.hparams.n_train_steps
        )
        return [optimizer], [
            {"scheduler": scheduler, "interval": "step", "frequency": 1}
        ]
