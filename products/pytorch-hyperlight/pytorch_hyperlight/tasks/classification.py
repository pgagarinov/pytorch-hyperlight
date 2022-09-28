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
import torchmetrics as metrics
import torch
from abc import abstractmethod
from transformers import AdamW, get_linear_schedule_with_warmup
from torch import nn, optim
import torch.nn.functional as F


class LitMetricsCalc(torch.nn.Module):
    def __init__(self, prefix, num_classes, metric_list=None):
        super(LitMetricsCalc, self).__init__()
        if metric_list is None:
            metric_list = ["acc", "f1"]

        metrics_dict = nn.ModuleDict()
        if "acc" in metric_list:
            metrics_dict["acc"] = metrics.classification.Accuracy()
        if "f1" in metric_list:
            metrics_dict["f1"] = metrics.classification.F1Score(
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
            metric(probs, target.long())

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

    def forward(self, *args, **kwargs):

        logits = self.model(*args, **kwargs)

        return logits

    def _forward_batch(self, batch):
        inputs = batch[0]
        return self(inputs)

    @staticmethod
    def _get_target_from_batch(batch):
        return batch[1]

    def __calc_loss(self, logits, target, log_label):
        loss = self.criterion(logits, target)
        self.log(f"{log_label}_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    # noinspection PyUnusedLocal
    def __stage_step(self, metric_calc, batch, batch_idx, stage):
        logits = self._forward_batch(batch)
        target = self._get_target_from_batch(batch)
        mval_dict = metric_calc.step(logits, target)
        self.log_dict(mval_dict, prog_bar=True, on_step=True, on_epoch=True)
        loss = self.__calc_loss(logits, target, stage)
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


class FCClassifier(nn.Module):
    def __init__(self, n_features, n_classes, dropout):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.classifier0 = nn.Linear(n_features, n_features)
        self.classifier1 = nn.Linear(n_features, n_classes)
        #
        self.classifier0.weight.data.normal_(mean=0.0, std=0.02)
        self.classifier0.bias.data.zero_()
        self.classifier1.weight.data.normal_(mean=0.0, std=0.02)
        self.classifier1.bias.data.zero_()

    def forward(self, pooled_output):

        pooled_output = self.classifier0(pooled_output)
        pooled_output = F.relu(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier1(pooled_output)
        return logits


class AAutoClsHeadClassificationWeightDecayTask(AClassificationTask):
    def __init__(self, hparams, model, criterion):
        classifier_module_name = self._get_classifier_module_name()
        old_classifier = getattr(model, classifier_module_name)
        new_classifier = self._build_classifier(hparams, old_classifier.in_features)
        setattr(model, classifier_module_name, new_classifier)
        super().__init__(hparams, model, criterion)

    @abstractmethod
    def _get_classifier_module_name(self):
        pass

    def _build_classifier(self, hparams, n_features):
        n_classes = hparams["n_classes"]
        fc_dropout = hparams["classifier_dropout"]
        fc = FCClassifier(n_features, n_classes, fc_dropout)
        return fc

    def get_param_groups(self):
        classifier_prefix = self._get_classifier_module_name() + "."

        classifier_param_list = [
            param
            for name, param in self.model.named_parameters()
            if name.startswith(classifier_prefix)
        ]
        assert len(classifier_param_list) > 0

        no_weight_decay_param_names = self.hparams["no_weight_decay_param_names"]

        rest_no_decay_param_list = [
            param
            for name, param in self.model.named_parameters()
            if not name.startswith(classifier_prefix)
            and any(nd in name for nd in no_weight_decay_param_names)
        ]

        rest_decay_param_list = [
            param
            for name, param in self.model.named_parameters()
            if not name.startswith(classifier_prefix)
            and (not any(nd in name for nd in no_weight_decay_param_names))
        ]

        return {
            "classifier": classifier_param_list,
            "rest_no_decay": rest_no_decay_param_list,
            "rest_decay": rest_decay_param_list,
        }

    def get_optimizer_params(self):
        param_groups_dict = self.get_param_groups()

        opt_arg_list = []

        for param_group_name, param_list in param_groups_dict.items():
            param_group_prefix = param_group_name.split("_")[0]
            lr_param_name = f"{param_group_prefix}_lr"
            if param_group_name == "rest_decay":
                weight_decay = self.hparams["weight_decay"]
            else:
                weight_decay = 0

            opt_arg_list.append(
                {
                    "params": param_list,
                    "lr": self.hparams[lr_param_name],
                    "weight_decay": weight_decay,
                }
            )

        return opt_arg_list


class AAutoClsHeadClassificationTaskWDAdamWStepLR(
    AAutoClsHeadClassificationWeightDecayTask
):
    def configure_optimizers(self):
        opt_arg_list = self.get_optimizer_params()

        optimizer = optim.AdamW(opt_arg_list)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.hparams.step_lr_step_size,
            gamma=self.hparams.step_lr_gamma,
        )
        return [optimizer], [scheduler]


class AAutoClsHeadClassificationTaskWDAdamWWarmup(
    AAutoClsHeadClassificationWeightDecayTask
):
    def configure_optimizers(self):
        opt_arg_list = self.get_optimizer_params()
        optimizer = AdamW(
            opt_arg_list,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer, self.hparams.warmup, self.hparams.n_train_steps
        )

        return [optimizer], [
            {"scheduler": scheduler, "interval": "step", "frequency": 1}
        ]
