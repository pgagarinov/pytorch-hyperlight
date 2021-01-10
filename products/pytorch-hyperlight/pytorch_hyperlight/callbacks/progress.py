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

from pytorch_lightning.callbacks import (
    ProgressBar,
)
from pytorch_hyperlight.utils.metric_dict_utils import MetricDictUtils
from tabulate import tabulate
from tqdm.autonotebook import tqdm


class LoggingProgressBar(ProgressBar):
    # noinspection PyUnusedLocal
    @staticmethod
    def __default_name_metric_transform(stage_list, metric_name):
        # stage can be 'train', 'validation' and 'test'
        return metric_name

    @staticmethod
    def __default_name_stage_transform(stage_list):
        # stage can be 'train', 'val' and 'test'
        return stage_list

    def __get_transformed_stage_name_pretty(self, stage_list):
        # stage can be 'train', 'val' and 'test'
        stage_list = self.__f_name_stage_transform(stage_list)
        return MetricDictUtils.get_stage_name_pretty(stage_list), stage_list

    def __init__(
        self,
        *args,
        f_name_stage_transform=None,
        f_name_metric_transform=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.__metrics_df_list = []

        if not f_name_stage_transform:
            f_name_stage_transform = LoggingProgressBar.__default_name_stage_transform
        if not f_name_metric_transform:
            f_name_metric_transform = LoggingProgressBar.__default_name_metric_transform

        self.__f_name_stage_transform = f_name_stage_transform
        self.__f_name_metric_transform = f_name_metric_transform

    def set_name_stage_transform(self, f_name_stage_transform):
        self.__f_name_stage_transform = f_name_stage_transform

    def set_name_metric_transform(self, f_name_metric_transform):
        self.__f_name_metric_transform = f_name_metric_transform

    def get_name_stage_transform(self):
        return self.__f_name_stage_transform

    def get_name_metric_transform(self):
        return self.__f_name_metric_transform

    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.leave = False
        return bar

    def init_test_tqdm(self):
        STAGE_LIST = ["test"]
        bar = super().init_test_tqdm()
        bar.leave = False
        stage_name_pretty = self.__get_transformed_stage_name_pretty(STAGE_LIST)[0]
        bar.set_description(stage_name_pretty)
        return bar

    def init_validation_tqdm(self):
        STAGE_LIST = ["val"]
        bar = super().init_validation_tqdm()
        stage_name_pretty = self.__get_transformed_stage_name_pretty(STAGE_LIST)[0]
        bar.set_description(stage_name_pretty)
        return bar

    def __log(self, stage_list, trainer):
        metrics_dict = MetricDictUtils.filter_n_round_epoch_metrics(
            trainer.progress_bar_dict
        )

        metric_prefix_set = MetricDictUtils.get_prefix_set(metrics_dict)

        ptl_stage_list = [p for p in stage_list if p in metric_prefix_set]

        metrics_dict = {
            self.__f_name_metric_transform(ptl_stage_list, k): v
            for k, v in metrics_dict.items()
        }

        stage_list = self.__f_name_stage_transform(ptl_stage_list)
        epoch = trainer.current_epoch

        metrics_df = MetricDictUtils.metrics_dict2df(metrics_dict, epoch, stage_list)
        self.__metrics_df_list.append(metrics_df)

        metrics_table_str = tabulate(
            metrics_df.drop(columns=["stage-list"]), headers="keys", tablefmt="pipe"
        )
        tqdm.write(metrics_table_str)

    def get_metrics_df(self):
        return MetricDictUtils.metrics_df_list_concat(self.__metrics_df_list)

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
