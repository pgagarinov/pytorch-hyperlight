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

import pandas as pd
from pytorch_hyperlight.metrics import TrialMetrics
from IPython.display import display
from copy import deepcopy


class InMemMetricStorage:
    def __init__(self):
        self.__metric_df_dict = {}

    def log_trial_metrics(self, trial_metrics, trial_name):
        trial_name.replace("_", "-")
        assert (
            trial_name not in self.__metric_df_dict
        ), f"metrics for run {trial_name} have already been logged"
        self.__metric_df_dict[trial_name] = trial_metrics

    def get_run_names(self):
        return set(self.__metric_df_dict.keys())

    def get_metrics(self, **kwargs):
        if len(self.__metric_df_dict) > 0:
            metrics_dict = deepcopy(self.__metric_df_dict)
            all_df = self.combine_exp_metric_df(metrics_dict, **kwargs)
            run_x_last_metric_df = self.combine_exp_metric_ser(metrics_dict)

            all_df = all_df.reset_index()
            epoch_x_stage_run_metric = TrialMetrics(all_df)
        else:
            run_x_last_metric_df = pd.DataFrame()
            epoch_x_stage_run_metric = TrialMetrics.create_empty()

        return {
            "run_x_last_metric_df": run_x_last_metric_df,
            "epoch_x_stage_run_metric": epoch_x_stage_run_metric,
        }

    @staticmethod
    def combine_exp_metric_df(metrics_dict, keep_train_val_only=False):
        df_dict = {k: v.df for k, v in metrics_dict.items()}
        df_list = [
            xdf.rename(
                columns={
                    col: col.replace("_", "-" + mod.replace("_", "-") + "_", 1)
                    for col in xdf.columns
                }
            )
            for mod, xdf in df_dict.items()
        ]
        df_list = [
            xdf.set_index(TrialMetrics.ALL_INDEX_COLUMN_LIST, drop=True)
            for xdf in df_list
        ]
        #
        #  in the following statement on=TrialMetrics.ALL_INDEX_COLUMN_LIST is a workaround for
        #  https://github.com/pandas-dev/pandas/issues/39100
        df_all = df_list[0]
        for cur_df in df_list[1:]:
            df_all = df_all.merge(
                cur_df, how="outer", on=TrialMetrics.ALL_INDEX_COLUMN_LIST
            )

        if keep_train_val_only:
            needed_cols = [
                c for c in df_all.columns if c[:5] != "reval" and c[:4] != "test"
            ]
            df_all = df_all[needed_cols]

        return df_all

    @staticmethod
    def combine_exp_metric_ser(results_dict):
        ser_dict = {k: v.series_last for k, v in results_dict.items()}
        run_list = list(ser_dict.keys())
        for run in run_list:
            ser_dict[run]["run"] = run
        ser_list = [v.to_frame().T for k, v in ser_dict.items()]
        df_last = pd.concat(ser_list)
        return df_last

    def show_report(self, sort_by_metric_list=None, figsize=(20, 10), **kwargs):
        if len(self.__metric_df_dict) > 0:
            metrics_dict = self.get_metrics(**kwargs)
            run_x_last_metric_df = metrics_dict["run_x_last_metric_df"]
            if sort_by_metric_list is None:
                sort_by_metric_list = list(run_x_last_metric_df.columns)
            else:
                sort_by_metric_list = sort_by_metric_list + ["run"]
            run_x_last_metric_df = run_x_last_metric_df.loc[
                :, sort_by_metric_list
            ].sort_values(sort_by_metric_list, ascending=False)
            display(run_x_last_metric_df)
            metrics_dict["epoch_x_stage_run_metric"].plot(figsize=figsize)
