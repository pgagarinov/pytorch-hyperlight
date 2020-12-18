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
    def __default_name_metric_pretty(stage_list, metric_name):
        # stage can be 'train', 'validation' and 'test'
        return metric_name

    @staticmethod
    def __default_name_stage_pretty(stage_list):
        # stage can be 'train', 'val' and 'test'
        return "/".join([s.capitalize() for s in stage_list])

    def __init__(
        self,
        *args,
        f_name_stage_pretty=None,
        f_name_metric_pretty=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.__metrics_df_list = []

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
        bar = super().init_train_tqdm()
        bar.leave = False
        return bar

    def init_test_tqdm(self):
        STAGE_LIST = ["test"]
        bar = super().init_test_tqdm()
        bar.leave = False
        bar.set_description(self.__f_name_stage_pretty(STAGE_LIST))
        return bar

    def init_validation_tqdm(self):
        STAGE_LIST = ["val"]
        bar = super().init_validation_tqdm()
        bar.set_description(self.__f_name_stage_pretty(STAGE_LIST))
        return bar

    @staticmethod
    def __metrics_dict2df(metrics_dict, epoch, stage_name):
        # sort metric names in alphabetical order from the tail
        sorted_metric_list = [
            e[::-1] for e in sorted([e[::-1] for e in list(metrics_dict.keys())])
        ]
        #
        metrics_df = pd.DataFrame(
            metrics_dict, index=[epoch], columns=sorted_metric_list
        )
        metrics_df.index.name = stage_name
        #
        return metrics_df

    def __log(self, stage_list, trainer):
        metrics_dict = self.__filter_metrics(trainer.progress_bar_dict)

        metric_prefix_set = MetricDictUtils.get_prefix_set(metrics_dict)

        stage_list = [p for p in stage_list if p in metric_prefix_set]

        metrics_dict = {
            self.__f_name_metric_pretty(stage_list, k): v
            for k, v in metrics_dict.items()
        }
        stage_name_pretty = self.__f_name_stage_pretty(stage_list)
        epoch = trainer.current_epoch
        #
        metrics_df = self.__metrics_dict2df(metrics_dict, epoch, stage_name_pretty)
        self.__metrics_df_list.append(metrics_df)
        #
        metrics_table_str = tabulate(metrics_df, headers="keys", tablefmt="pipe")
        tqdm.write(metrics_table_str)

    def get_metrics_df(self):
        metrics_df_list = self.__metrics_df_list
        epoch_list = [e.index[0] for e in metrics_df_list]
        stage_list = [e.index.name for e in metrics_df_list]
        df = pd.concat(metrics_df_list)
        df.insert(loc=0, column='stage', value=stage_list)
        df.insert(loc=0, column='epoch', value=epoch_list)
        df.reset_index(inplace=True, drop=True)
        return df

    def on_sanity_check_end(self, trainer, pl_module):
        pass

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        STAGE_LIST = ["train", "val"]
        super().on_train_epoch_end(trainer, pl_module, outputs)
        self.main_progress_bar.close()
        if not trainer.running_sanity_check:
            self.__log(STAGE_LIST, trainer)
        self.main_progress_bar = self.init_train_tqdm()

    def on_validation_epoch_end(self, trainer, pl_module):
        pass

    def on_test_epoch_end(self, trainer, pl_module):
        STAGE_LIST = ["test"]
        super().on_test_epoch_end(trainer, pl_module)
        if not trainer.running_sanity_check:
            self.__log(STAGE_LIST, trainer)
