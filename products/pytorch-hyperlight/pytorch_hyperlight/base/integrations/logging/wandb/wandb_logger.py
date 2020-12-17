from pathlib import Path
from pytorch_hyperlight.base.utils.experiment_trial_namer import ExperimentTrialNamer
import logging
import os

# noinspection PyUnresolvedReferences
class WandBIntegrator:
    def __init__(self, experiment_id, silent=True):
        self.__experiment_id = experiment_id
        self.set_key_if_exists()
        if silent:
            self.make_silent()

    @staticmethod
    def __disable_warnings():
        logger = logging.getLogger("wandb")
        logger.setLevel(logging.ERROR)

    @staticmethod
    def get_key_path():
        key_path = Path.home() / ".wandb_api_key"
        return key_path

    def make_silent(self):
        os.environ["WANDB_SILENT"] = "true"
        self.__disable_warnings()

    def get_key(self):
        return self.get_key_path().read_text().replace("\n", "")

    def key_exists(self):
        return self.get_key_path().exists()

    def set_key_if_exists(self):
        if self.key_exists():
            os.environ["WANDB_API_KEY"] = self.get_key()
        else:
            # warnings.warn('WandB key not found')
            # print('WandB key not found')
            logging.getLogger(self.set_key_if_exists.__name__).warning(
                "WandB key not found"
            )

    def configure_raytune(self, exp_config):
        if self.key_exists():
            exp_config = exp_config.copy()
            exp_config["wandb"] = {
                "project": self.__experiment_id,
                "group": ExperimentTrialNamer.get_group_name(),
                "api_key": self.get_key(),
                "log_config": True,
            }
        return exp_config

    def get_pl_loggers(self):
        if self.key_exists():
            pl_loggers = [
                pl.loggers.wandb.WandbLogger(
                    project=self.__experiment_id, name=self.get_group(), group="manual"
                )
            ]
        else:
            pl_loggers = []
        return pl_loggers

    def get_raytune_loggers(self):
        if self.key_exists():
            raytune_loggers = [tune.integration.wandb.WandbLogger]
        else:
            raytune_loggers = []

        return raytune_loggers