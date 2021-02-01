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

import os
from functools import partial

import pytorch_lightning as pl
from ray import tune

from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)

from pytorch_hyperlight.callbacks.early_stopping import EarlyStoppingWithGracePeriod
from pytorch_lightning.utilities.cloud_io import load as pl_load

from ray.tune.integration.pytorch_lightning import (  # TuneReportCallback,
    TuneReportCheckpointCallback,
)

from pytorch_hyperlight.runner.raytune_runner import (
    run_tune_experiment_asha_hyperopt,
    tune_init,
)
from pytorch_hyperlight.integrations.logging.wandb.wandb_logger import (
    WandBIntegrator,
    DummyWandBIntegrator,
)
from pytorch_hyperlight.utils.experiment_trial_namer import ExperimentTrialNamer
from pytorch_hyperlight.callbacks.progress import LoggingProgressBar
from pytorch_hyperlight.utils.metric_dict_utils import MetricDictUtils
from pytorch_hyperlight.metrics.trial_metrics import TrialMetrics

#
import pandas as pd


class LitModuleBuilder:
    def __init__(self, model_class):
        self.__model_class = model_class

    def load_from_checkpoint(self, checkpoint_path, **kwargs):
        (
            lmodule,
            last_epoch,
            _,
        ) = LitModuleBuilder.__load_lmodule_from_checkpoint_with_extra_details(
            self.__model_class, checkpoint_path, **kwargs
        )
        return lmodule, last_epoch

    def create(self, hparams):
        lmodule = self.__model_class(hparams=hparams)
        return lmodule

    # noinspection PyProtectedMember
    @staticmethod
    def __load_lmodule_from_checkpoint_with_extra_details(
        model_class, checkpoint_path, **kwargs
    ):
        """
        We use this custom function instead of the built-in `load_from_checkpoint` function
        from PyTorch Lightning to make it possible to load best_epoch, not just the best model
        """
        ckpt = pl_load(
            checkpoint_path,
            map_location=lambda storage, loc: storage,
        )
        epoch = ckpt["epoch"]
        # use strict=False by default
        lmodule = model_class._load_model_state(ckpt, strict=False, **kwargs)
        #
        return lmodule, epoch, ckpt


class BaseRunner:
    def __init__(
        self,
        f_configure_dataloaders,
        pl_callbacks=None,
        pl_loggers=None,
        is_debug=False,
        experiment_id=None,
        log2wandb=False,
    ):

        if pl_callbacks is None:
            pl_callbacks = []
        if pl_loggers is None:
            pl_loggers = []
        self.__experiment_id = experiment_id
        #
        if log2wandb:
            self.wandb_integrator = WandBIntegrator(self.__experiment_id)
        else:
            self.wandb_integrator = DummyWandBIntegrator()
        #
        pl_loggers = self.__process_pl_loggers_hook(pl_loggers)
        self.__pl_loggers = pl_loggers
        #
        self.__raytune_loggers = []
        self.__raytune_loggers = self.__process_raytune_loggers_hook(
            self.__raytune_loggers
        )
        #
        self.__pl_callbacks = pl_callbacks
        self.__f_configure_dataloaders = f_configure_dataloaders

        self.__is_debug = is_debug

    def __process_pl_loggers_hook(self, pl_loggers):
        pl_loggers = pl_loggers.copy()
        pl_loggers.extend(self.wandb_integrator.get_pl_loggers())
        return pl_loggers

    def __process_search_space_hook(self, search_space_config):
        search_space_config = self.wandb_integrator.configure_raytune(
            search_space_config
        )
        return search_space_config

    def __process_raytune_loggers_hook(self, raytune_loggers):
        raytune_loggers = raytune_loggers.copy()
        raytune_loggers.extend(self.wandb_integrator.get_raytune_loggers())
        return raytune_loggers

    def __get_dataloaders(self, batch_size, n_workers):
        return self.__f_configure_dataloaders(batch_size, n_workers=n_workers)

    def __get_pl_callbacks_extended(self):
        pl_callbacks = self.__pl_callbacks.copy()
        lprogress_bar_callback = LoggingProgressBar()
        pl_callbacks.append(lprogress_bar_callback)
        return pl_callbacks, lprogress_bar_callback

    def run_single_trial(self, lmodule_class, config, extra_config):

        pl_callbacks, lprogress_bar_callback = self.__get_pl_callbacks_extended()

        extra_config = self.__add_experiment_id_to_dict(extra_config)

        lmodule_builder = LitModuleBuilder(lmodule_class)

        train_result = self.__run_in_main_process(
            lmodule_builder,
            config,
            extra_config=extra_config,
            pl_callbacks=pl_callbacks,
            pl_loggers=self.__pl_loggers,
        )
        #
        # we do not need the last model state
        #
        del train_result["lmodule"]

        (lmodule_best, best_epoch) = lmodule_builder.load_from_checkpoint(
            train_result["trainer"].checkpoint_callback.best_model_path
        )
        train_val_metrics_dict = self.__get_metrics(train_result["trainer"])

        reval_test_metrics_dict = self.__revalidate_n_test_if_possible(
            lmodule_best,
            train_result["trainer"],
            extra_config,
            train_result["dataloaders_dict"],
            lprogress_bar_callback,
        )

        metrics_dict = {**train_val_metrics_dict, **reval_test_metrics_dict}

        trial_metrics = TrialMetrics.from_metrics_df_list(
            lprogress_bar_callback.get_metrics_df_list()
        )

        self.__check_trial_metrics(trial_metrics, metrics_dict)

        return {
            "lmodule_best": lmodule_best,
            "best_epoch": best_epoch,
            "trainer": train_result["trainer"],
            "metrics": trial_metrics,
        }

    @staticmethod
    def __check_trial_metrics(trial_metrics, metrics_dict):
        expected_series_last = pd.Series(
            MetricDictUtils.filter_n_round_epoch_metrics(metrics_dict)
        ).sort_index()
        assert trial_metrics.series_last.sort_index().equals(
            expected_series_last
        ), "Oops, metrics values are inconsistent"

    def __run_in_main_process(self, lmodule_builder, config, **kwargs):
        """
        Similar wrapper but this time for running single experiments in manual mode
        """
        return self.__train_with_checkpoint(
            config,
            self.__f_configure_dataloaders,
            lmodule_builder,
            checkpoint_dir=None,
            from_ray=False,
            is_debug=self.__is_debug,
            **kwargs,
        )

    def __add_experiment_id_to_dict(self, some_dict):
        some_dict = some_dict.copy()
        some_dict["experiment_id"] = (
            self.__experiment_id + "_" + ExperimentTrialNamer.get_group_name()
        )
        return some_dict

    def run_hyper_opt(
        self,
        lmodule_class,
        search_space_config,
        tune_config,
        f_tune_init=tune_init,
        f_run_tune_experiment=run_tune_experiment_asha_hyperopt,
    ):

        lmodule_builder = LitModuleBuilder(lmodule_class)

        f_tune_init()
        #
        tune_config = self.__add_experiment_id_to_dict(tune_config)
        #
        search_space_config = self.__process_search_space_hook(search_space_config)
        #
        f_run_tune_experiment = partial(
            f_run_tune_experiment, raytune_loggers=self.__raytune_loggers
        )
        #

        f_trainer = tune.with_parameters(
            self.__train_with_checkpoint_drop_output,
            f_configure_dataloaders=self.__f_configure_dataloaders,
            lmodule_builder=lmodule_builder,
            extra_config=tune_config,
            pl_callbacks=self.__pl_callbacks,
            pl_loggers=self.__pl_loggers,
            is_debug=self.__is_debug,
        )

        analysis = f_run_tune_experiment(
            f_trainer,
            search_space_config,
            tune_config,
        )

        loaders_dict = self.__get_dataloaders(
            tune_config["batch_size_main"], tune_config["cpu_per_trial"]
        )

        checkpoint_path = os.path.join(analysis.best_checkpoint, "checkpoint.ckpt")

        lmodule_best, best_epoch = lmodule_builder.load_from_checkpoint(checkpoint_path)

        pl_callbacks, lprogress_bar_callback = self.__get_pl_callbacks_extended()

        trainer = self.__create_trainer(
            analysis.best_config,
            tune_config,
            False,
            is_debug=self.__is_debug,
            pl_callbacks=pl_callbacks,
            pl_loggers=self.__pl_loggers,
        )

        #
        metrics_dict = self.__revalidate_n_test_if_possible(
            lmodule_best, trainer, tune_config, loaders_dict, lprogress_bar_callback
        )

        metrics_df_list = lprogress_bar_callback.get_metrics_df_list()
        for metrics_df in metrics_df_list:
            metrics_df.epoch = best_epoch

        trial_metrics = TrialMetrics.from_metrics_df_list(metrics_df_list)

        self.__check_trial_metrics(trial_metrics, metrics_dict)

        return {
            "lmodule_best": lmodule_best,
            "best_epoch": best_epoch,
            "metrics": trial_metrics,
            "analysis": analysis,
        }

    def __revalidate(
        self,
        lmodule_best,
        trainer,
        extra_config,
        val_dataloader,
        lprogress_bar_callback,
    ):
        def __test_is_reval_name_metric_transform(stage_list, metric_name):
            # stage can be 'train', 'validation' and 'test'
            assert stage_list == ["test"]
            metric_name = metric_name.replace("test_", "reval_")

            return metric_name

        def __test_is_reval_name_stage_transform(stage_list):
            # stage can be 'train', 'validation' and 'test'
            assert stage_list == ["test"]
            return ["reval"]

        #
        prev_name_metric_transform = lprogress_bar_callback.get_name_metric_transform()
        prev_name_stage_transform = lprogress_bar_callback.get_name_stage_transform()
        #
        lprogress_bar_callback.set_name_metric_transform(
            __test_is_reval_name_metric_transform
        )
        lprogress_bar_callback.set_name_stage_transform(
            __test_is_reval_name_stage_transform
        )

        self.__set_seed(extra_config)
        val_result = trainer.test(
            lmodule_best, test_dataloaders=val_dataloader, verbose=False
        )

        lprogress_bar_callback.set_name_metric_transform(prev_name_metric_transform)
        lprogress_bar_callback.set_name_stage_transform(prev_name_stage_transform)

        reval_metrics_dict = MetricDictUtils.filter_by_suffix(val_result[0], "_epoch")
        reval_metrics_dict = MetricDictUtils.change_prefix(
            reval_metrics_dict, "test_", "reval_"
        )
        return reval_metrics_dict

    def __test(self, lmodule_best, trainer, extra_config, test_dataloader):
        self.__set_seed(extra_config)
        val_result = trainer.test(
            lmodule_best, test_dataloaders=test_dataloader, verbose=False
        )

        reval_metrics_dict = MetricDictUtils.filter_by_suffix(val_result[0], "_epoch")
        return reval_metrics_dict

    def __revalidate_n_test_if_possible(
        self, lmodule_best, trainer, extra_config, loaders_dict, lprogress_bar_callback
    ):
        reval_metrics_dict = {}
        if "val_loader_name" in extra_config:
            val_loader_name = extra_config["val_loader_name"]
            if val_loader_name:
                reval_metrics_dict = self.__revalidate(
                    lmodule_best,
                    trainer,
                    extra_config,
                    loaders_dict[val_loader_name],
                    lprogress_bar_callback,
                )

        test_metrics_dict = {}
        if "test_loader_name" in extra_config:
            test_loader_name = extra_config["test_loader_name"]
            if test_loader_name:
                test_metrics_dict = self.__test(
                    lmodule_best,
                    trainer,
                    extra_config,
                    loaders_dict[test_loader_name],
                )

        metrics_dict = {**reval_metrics_dict, **test_metrics_dict}
        return metrics_dict

    @staticmethod
    def __get_metrics(trainer):
        metrics_dict = MetricDictUtils.strip_tensors(trainer.callback_metrics)
        metrics_dict = MetricDictUtils.filter_by_suffix(metrics_dict, "_epoch")
        return metrics_dict

    @staticmethod
    def __create_trainer(
        config,
        extra_config,
        from_ray,
        is_debug=False,
        pl_callbacks=None,
        pl_loggers=None,
    ):
        if pl_callbacks is None:
            pl_callbacks = []
        if pl_loggers is None:
            pl_loggers = []
        #
        config = config.copy()
        #
        if is_debug:
            TRAINER_KWARG_DICT = {
                "max_epochs": 3,
                "limit_train_batches": 10,
                "limit_val_batches": 5,
                "limit_test_batches": 5,
            }
        else:
            TRAINER_KWARG_DICT = {"max_epochs": config["max_epochs"]}

        #
        lr_monitor = LearningRateMonitor(logging_interval="step")
        pl_callbacks.append(lr_monitor)
        #
        if "ptl_early_stopping_patience" in extra_config:
            if "grace_period" in extra_config:
                es_kwargs = {"grace_period": extra_config["grace_period"]}
            else:
                es_kwargs = {}
            #
            es_callback = EarlyStoppingWithGracePeriod(
                monitor=extra_config["metric_to_optimize"],
                patience=extra_config["ptl_early_stopping_patience"],
                verbose=True,
                mode=extra_config["metric_opt_mode"],
                min_delta=0,
                **es_kwargs,
            )
            pl_callbacks.append(es_callback)
        #
        if from_ray:
            metric_list = extra_config["ray_metrics_to_show"]
            tune_val_callback = TuneReportCheckpointCallback(
                metrics=dict(zip(metric_list, metric_list)),
                filename="checkpoint.ckpt",
                on="validation_end",
            )
            pl_callbacks.append(tune_val_callback)

            TRAINER_KWARG_DICT["progress_bar_refresh_rate"] = 0
            TRAINER_KWARG_DICT["checkpoint_callback"] = False
            TRAINER_KWARG_DICT["weights_summary"] = None
        else:
            PTL_CHECKPOINT_PERIOD = 1
            PTL_CHECKPOINT_SAVE_TOP_K = 2
            #
            checkpoint_callback = ModelCheckpoint(
                monitor=extra_config["metric_to_optimize"],
                mode=extra_config["metric_opt_mode"],
                save_top_k=PTL_CHECKPOINT_SAVE_TOP_K,
                period=PTL_CHECKPOINT_PERIOD,
            )
            pl_callbacks.append(checkpoint_callback)

        # It is yet to be figured out hot to report training metrics
        # (currently only validations metrics are reported).
        # The following line leads to errors
        """
        metric_list =['train_acc_epoch', 'train_loss_epoch']
        tune_train_callback = TuneReportCallback(
            metrics=dict(zip(metric_list, metric_list)),
            on="train_end",
        )
        """
        #
        is_trainer_determenistic = "seed" in extra_config
        trainer = pl.Trainer(
            deterministic=is_trainer_determenistic,
            gpus=extra_config["gpus"],
            check_val_every_n_epoch=1,
            # progress_bar_refresh_rate = 0,
            gradient_clip_val=config["gradient_clip_val"],
            precision=extra_config["ptl_precision"],
            callbacks=pl_callbacks,
            logger=pl_loggers,
            **TRAINER_KWARG_DICT,
        )

        return trainer

    @staticmethod
    def __set_seed(extra_config):
        if "seed" in extra_config:
            pl.seed_everything(extra_config["seed"])

    @staticmethod
    def __train_with_checkpoint(
        config,
        f_configure_dataloaders,
        lmodule_builder,
        checkpoint_dir=None,
        from_ray=True,
        is_debug=False,
        extra_config=None,
        pl_callbacks=None,
        pl_loggers=None,
    ):
        """
        This is a wrapper around PyTorch Lightning Trainer to make PTL and Ray Tune best friends
        (see https://docs.ray.io/en/master/tune/tutorials/tune-pytorch-lightning.html).
        The hyper parameter optimization algorithm run by Ray Tune may decide to restore
        the model from the checkpoint in which case checkpoint_dir is not None.
        Such situation is possible when Population Based Training in Ray Tune is used.
        The parameters are as follows:

        - **config** contains a current set of hyperparameter values
        - **f_configure_dataloaders** is a function used for generating dataloaders
        - **checkpoint_dir** points to the checkpoint directory in case Ray Tune expects
            the model to be restored from the checkpoint
        - **from_ray**, if False, allows to run this function manually i.e. without Ray Tune.
            This is used for a single-experiment train-validate cycle.
            If False, the function is expected by be launched by Ray Tune in automatic
            mode of hyper-parameters search
        - **extra_config** contains the experiment parameters (including the technical
            ones such as number of CPUs per experiment) that are not expected to be tuned
            as hyper-parameters
        """

        trainer = BaseRunner.__create_trainer(
            config,
            extra_config,
            from_ray,
            is_debug=is_debug,
            pl_callbacks=pl_callbacks,
            pl_loggers=pl_loggers,
        )

        config = config.copy()
        #
        # ----!!!----SET SEED TO MAKE EXPERIMENTS DETERMENISTIC----!!!----[FIRST TIME]
        loaders_dict = f_configure_dataloaders(
            config["batch_size"], n_workers=extra_config["cpu_per_trial"]
        )
        #
        train_loader_name = extra_config["train_loader_name"]
        train_loader = loaders_dict[train_loader_name]
        #
        config["n_steps_per_epoch"] = len(train_loader)
        config["n_train_steps"] = len(train_loader) * config["max_epochs"]
        #
        if checkpoint_dir:
            #
            # Here we restore state of things and plan to continue
            # training with new hyper-parameters
            #
            checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.ckpt")

            lmodule, last_epoch = lmodule_builder.load_from_checkpoint(
                checkpoint_path, hparams=config
            )

            trainer.current_epoch = last_epoch
            #
        else:
            # ----!!!----SET SEED TO MAKE EXPERIMENTS DETERMENISTIC----!!!----[SECOND TIME]
            BaseRunner.__set_seed(extra_config)
            lmodule = lmodule_builder.create(config)
        #
        fit_kwargs_dict = {}

        if "val_loader_name" in extra_config:
            val_loader_name = extra_config["val_loader_name"]
            if val_loader_name:
                fit_kwargs_dict["val_dataloaders"] = loaders_dict[val_loader_name]
        #
        # ----!!!----SET SEED TO MAKE EXPERIMENTS DETERMENISTIC----!!!----[THIRD TIME]
        BaseRunner.__set_seed(extra_config)
        _ = trainer.fit(lmodule, train_loader, **fit_kwargs_dict)

        return {
            "lmodule": lmodule,
            "trainer": trainer,
            "dataloaders_dict": loaders_dict,
        }

    @staticmethod
    def __train_with_checkpoint_drop_output(config, *args, **kwargs):
        """
        Ray Tune doesn't expect any output arguments from the trainer function (except for
        the metrics reported back to Ray Tune for the trial selection). Thus this wrapper is
        implemented to skip the outputs (used when running a single experiment in manual mode).
        Please note `from_ray=True` which indicates to `train_with_checkpoint` function that
        function is launched from Ray Tune
        """
        _ = BaseRunner.__train_with_checkpoint(config, *args, from_ray=True, **kwargs)
