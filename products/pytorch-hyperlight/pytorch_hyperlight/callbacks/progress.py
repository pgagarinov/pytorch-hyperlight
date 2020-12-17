from pytorch_lightning.callbacks import (
    ProgressBar,
)
from pytorch_hyperlight.utils.metric_dict_utils import MetricDictUtils
from tabulate import tabulate
import pandas as pd
from tqdm.autonotebook import tqdm


class LoggingProgressBar(ProgressBar):
    # noinspection PyUnusedLocal
    @staticmethod
    def __default_name_metric_pretty(stage, metric_name):
        # stage can be 'train', 'validation' and 'test'
        return metric_name

    @staticmethod
    def __default_name_stage_pretty(stage):
        # stage can be 'train', 'validation' and 'test'
        if stage == "train":
            stage_pretty = "Tr/Val"
        elif stage == "validation":
            stage_pretty = "Val"
        elif stage == "test":
            stage_pretty = "Tst"
        else:
            raise NameError
        return stage_pretty

    def __init__(
        self,
        *args,
        f_name_stage_pretty=None,
        f_name_metric_pretty=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if not f_name_stage_pretty:
            f_name_stage_pretty = LoggingProgressBar.__default_name_stage_pretty
        if not f_name_metric_pretty:
            f_name_metric_pretty = LoggingProgressBar.__default_name_metric_pretty

        self.__f_name_stage_pretty = f_name_stage_pretty
        self.__f_name_metric_pretty = f_name_metric_pretty

    def set_name_stage_pretty(self, f_name_stage_pretty):
        self.__f_name_stage_pretty = f_name_stage_pretty

    def set_name_metric_pretty(self, f_name_metric_pretty):
        self.__f_name_metric_pretty = f_name_metric_pretty

    def get_name_stage_pretty(self):
        return self.__f_name_stage_pretty

    def get_name_metric_pretty(self):
        return self.__f_name_metric_pretty

    @staticmethod
    def __filter_metrics(metrics_dict):
        N_DIGITS_AFTER_DOT = 4

        metrics_dict = MetricDictUtils.filter_by_suffix(metrics_dict, "_epoch")
        metrics_dict = MetricDictUtils.remove_suffix(metrics_dict, "_epoch")
        metrics_dict = MetricDictUtils.round_floats(metrics_dict, N_DIGITS_AFTER_DOT)
        return metrics_dict

    def init_train_tqdm(self):
        # STAGE_NAME = "train"
        bar = super().init_train_tqdm()
        bar.leave = False
        return bar

    def init_test_tqdm(self):
        STAGE_NAME = "test"
        bar = super().init_test_tqdm()
        bar.leave = False
        bar.set_description(self.__f_name_stage_pretty(STAGE_NAME))
        return bar

    def init_validation_tqdm(self):
        STAGE_NAME = "validation"
        bar = super().init_validation_tqdm()
        bar.set_description(self.__f_name_stage_pretty(STAGE_NAME))
        return bar

    @staticmethod
    def __disp_dict(metrics_dict, epoch, stage_name):
        # sort metric names in alphabetical order from the tail
        sorted_metric_list = [
            e[::-1] for e in sorted([e[::-1] for e in list(metrics_dict.keys())])
        ]
        #
        metrics_df = pd.DataFrame(
            metrics_dict, index=[epoch], columns=sorted_metric_list
        )
        metrics_df.index.name = stage_name
        metrics_table_str = tabulate(metrics_df, headers="keys", tablefmt="pipe")
        tqdm.write(metrics_table_str)

    def __log(self, stage, trainer):
        metrics_dict = self.__filter_metrics(trainer.progress_bar_dict)
        metrics_dict = {
            self.__f_name_metric_pretty(stage, k): v for k, v in metrics_dict.items()
        }
        stage_name_pretty = self.__f_name_stage_pretty(stage)
        epoch = trainer.current_epoch
        self.__disp_dict(metrics_dict, epoch, stage_name_pretty)

    def on_sanity_check_end(self, trainer, pl_module):
        pass

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        STAGE_NAME = "train"
        super().on_train_epoch_end(trainer, pl_module, outputs)
        self.main_progress_bar.close()
        if not trainer.running_sanity_check:
            self.__log(STAGE_NAME, trainer)
        self.main_progress_bar = self.init_train_tqdm()

    def on_validation_epoch_end(self, trainer, pl_module):
        pass

    def on_test_epoch_end(self, trainer, pl_module):
        STAGE_NAME = "test"
        super().on_test_epoch_end(trainer, pl_module)
        if not trainer.running_sanity_check:
            self.__log(STAGE_NAME, trainer)
