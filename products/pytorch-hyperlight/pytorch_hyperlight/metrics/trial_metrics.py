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
import math
import matplotlib.pyplot as plt
import itertools
from IPython.display import display
from pytorch_hyperlight.utils.metric_dict_utils import MetricDictUtils

TRAIN_SUFFIX = "train"
VAL_SUFFIX = "val"
REVAL_SUFFIX = "reval"
TEST_SUFFIX = "test"
MARKER_LIST = ("o", "d", "p", "*", "<", "^", "s", "+", "x", "1", "2", ">")


def get_stage_suffix(inp_str):
    if inp_str.startswith(TRAIN_SUFFIX):
        return TRAIN_SUFFIX
    if inp_str.startswith(REVAL_SUFFIX):
        return REVAL_SUFFIX
    if inp_str.startswith(VAL_SUFFIX):
        return VAL_SUFFIX
    if inp_str.startswith(TEST_SUFFIX):
        return TEST_SUFFIX
    return None


def get_group_name(inp_str, suffix):
    return inp_str[len(suffix) :].split("_", 1)[0]


def get_df_column_styles(df):

    marker_iter = itertools.cycle(MARKER_LIST)
    style_extra_list = [""] * len(df.columns)
    group_name_list = [""] * len(df.columns)
    for i_col, c in enumerate(df.columns):

        suffix = get_stage_suffix(c)
        if suffix == TRAIN_SUFFIX:
            style_extra_list[i_col] = "-"

        if suffix is not None:
            group_name_list[i_col] = get_group_name(c, suffix)

    group_markers_dict = {k: next(marker_iter) for k in set(group_name_list)}

    style_list = [""] * len(df.columns)

    for i_col, c in enumerate(df.columns):
        group_name = group_name_list[i_col]
        style_list[i_col] = (
            group_markers_dict[group_name] + "-" + style_extra_list[i_col]
        )

    return style_list


class TrialMetrics:
    PLOT_INDEX_COLUMN_LIST = ["stage", "epoch"]
    ALL_INDEX_COLUMN_LIST = PLOT_INDEX_COLUMN_LIST + ["stage-list"]

    def __init__(self, metrics_df):
        assert isinstance(metrics_df, pd.DataFrame)
        assert all(
            [col_name in metrics_df.columns for col_name in self.ALL_INDEX_COLUMN_LIST]
        )
        metrics_df = metrics_df.copy()
        metrics_df = metrics_df[
            pd.Index(self.ALL_INDEX_COLUMN_LIST).append(
                metrics_df.columns.drop(self.ALL_INDEX_COLUMN_LIST)
            )
        ]
        metrics_df["stage-list"] = metrics_df["stage-list"].apply(tuple)

        col_stage_list = MetricDictUtils.get_list_prefix_list(
            metrics_df.columns, split_symbol_list=["_", "-"]
        )
        for i_col, col in enumerate(metrics_df.columns):
            if col in self.ALL_INDEX_COLUMN_LIST:
                continue
            col_stage = col_stage_list[i_col]
            stage_list = (
                metrics_df.loc[~metrics_df[col].isna(), "stage-list"]
                .apply(set)
                .tolist()
            )

            bad_stage_list = [x for x in stage_list if col_stage not in x]

            assert (
                len(bad_stage_list) == 0
            ), f'column "{col}" related to stage "{col_stage}" contains not NaN values for stages {bad_stage_list}'

        self.__metrics_df = metrics_df

    @staticmethod
    def create_empty():
        metric_df = pd.DataFrame(columns=TrialMetrics.ALL_INDEX_COLUMN_LIST)
        return TrialMetrics(metric_df)

    def get_df(self):
        return self.__metrics_df.copy()

    df = property(get_df)

    def get_series_last(self):
        return (
            self.__metrics_df.loc[
                :, self.__metrics_df.columns.drop(self.ALL_INDEX_COLUMN_LIST)
            ]
            .ffill(axis=0)
            .iloc[-1, :]
            .copy()
        )

    series_last = property(get_series_last)

    @staticmethod
    def create_subplots(n_graphs, figsize=None, max_cols=2):
        SUBPLOT_WIDTH = 20
        SUBPLOT_HEIGHT = 12
        MAX_COLS = max_cols
        n_cols = min(n_graphs, MAX_COLS)
        n_rows = math.ceil(n_graphs / n_cols)
        if figsize is None:
            figsize = (SUBPLOT_WIDTH, SUBPLOT_HEIGHT * n_rows / n_cols)
        fig = plt.figure(figsize=figsize)
        ax_list = [None] * n_graphs
        for i_graph in range(n_graphs):
            ax_list[i_graph] = fig.add_subplot(n_rows, n_cols, i_graph + 1)
        return fig, ax_list

    def plot(self, **kwargs):
        cols2drop_list = list(
            set(self.ALL_INDEX_COLUMN_LIST) - set(self.PLOT_INDEX_COLUMN_LIST)
        )
        df = self.__metrics_df.set_index(self.PLOT_INDEX_COLUMN_LIST).drop(
            columns=cols2drop_list
        )
        stage_metric_pair_list = [x.split("_", 1) for x in df.columns]
        new_column_tuples = [
            a[::-1] + [b] for a, b in zip(stage_metric_pair_list, df.columns)
        ]
        index = pd.MultiIndex.from_tuples(
            new_column_tuples, names=["metric_name", "metric_stage", "metric"]
        )
        df.columns = index

        metric_name_list = list(set(df.columns.get_level_values(0)))
        n_metrics = len(metric_name_list)

        fig, ax_list = self.create_subplots(n_metrics, **kwargs)

        for i_metric, ax in enumerate(ax_list):
            metric_name = metric_name_list[i_metric]
            metric_df = df.loc[:, metric_name]
            metric_df.columns = metric_df.columns.droplevel(0)
            metric_df.index = metric_df.index.droplevel(0)
            style_list = get_df_column_styles(metric_df)
            self.plot_df_with_dropped_nans(metric_df, ax, style_list, ms=5, grid=True)
        return fig, ax_list

    @staticmethod
    def from_metrics_df_list(metrics_df_list):
        return TrialMetrics(MetricDictUtils.metrics_df_list_concat(metrics_df_list))

    @staticmethod
    def plot_df_with_dropped_nans(df, ax, style_list, **kwargs):
        for i_col, col in enumerate(df.columns):
            col_series = df[col].dropna()
            col_series.plot(label=col, style=style_list[i_col], ax=ax, **kwargs)
        ax.legend()

    def show_report(self, **kwargs):
        display(self.df.drop(columns=["stage-list"]))
        self.plot(**kwargs)
