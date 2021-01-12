from pathlib import Path
from pytorch_hyperlight.utils.experiment_trial_namer import ExperimentTrialNamer
import logging
import os
import pytorch_lightning as pl
import ray.tune.integration.wandb as rtwb
from abc import ABC, abstractmethod


class IWandBIntegrator(ABC):
    @abstractmethod
    def configure_raytune(self, exp_config):
        pass

    @abstractmethod
    def get_pl_loggers(self):
        pass

    @abstractmethod
    def get_raytune_loggers(self):
        pass


# noinspection PyUnresolvedReferences
class WandBIntegrator(IWandBIntegrator):
    def __init__(self, experiment_id, silent=True):
        self.__experiment_id = experiment_id
        self.__set_key_if_exists()
        if silent:
            self.__make_silent()

    @staticmethod
    def __disable_warnings():
        logger = logging.getLogger("wandb")
        logger.setLevel(logging.ERROR)

    @staticmethod
    def __get_key_path():
        key_path = Path.home() / ".wandb_api_key"
        return key_path

    def __make_silent(self):
        os.environ["WANDB_SILENT"] = "true"
        self.__disable_warnings()

    @staticmethod
    def __get_group():
        return ExperimentTrialNamer.get_group_name()

    def __get_key(self):
        return self.__get_key_path().read_text().replace("\n", "")

    def __key_exists(self):
        return self.__get_key_path().exists()

    def __set_key_if_exists(self):
        if self.__key_exists():
            os.environ["WANDB_API_KEY"] = self.__get_key()
        else:
            # warnings.warn('WandB key not found')
            # print('WandB key not found')
            logging.getLogger(self.__set_key_if_exists.__name__).warning(
                "WandB key not found"
            )

    def configure_raytune(self, exp_config):
        if self.__key_exists():
            exp_config = exp_config.copy()
            exp_config["wandb"] = {
                "project": self.__experiment_id,
                "group": self.__get_group(),
                "api_key": self.__get_key(),
                "log_config": True,
            }
        return exp_config

    def get_pl_loggers(self):
        if self.__key_exists():
            pl_loggers = [
                pl.loggers.wandb.WandbLogger(
                    project=self.__experiment_id,
                    name=self.__get_group(),
                    group="manual",
                )
            ]
        else:
            pl_loggers = []
        return pl_loggers

    def get_raytune_loggers(self):
        if self.__key_exists():
            raytune_loggers = [rtwb.WandbLogger]
        else:
            raytune_loggers = []

        return raytune_loggers


class DummyWandBIntegrator(IWandBIntegrator):
    def __init__(self, *args, **kwargs):
        pass

    def configure_raytune(self, exp_config):
        return exp_config

    def get_pl_loggers(self):
        pl_loggers = []
        return pl_loggers

    def get_raytune_loggers(self):
        raytune_loggers = []
        return raytune_loggers
