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

TRAIN_SUFFIX = "train"
VAL_SUFFIX = "val"
REVAL_SUFFIX = "reval"
TEST_SUFFIX = "test"
MARKER_LIST = ("o", "+", "p", "<", "*", "^", "s", "d", "x", "1", "2", ">")


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
    def __init__(self, metrics_last_ser, metrics_df):
        self.__metrics_df = metrics_df
        self.__metrics_last_ser = metrics_last_ser

    def get_df(self):
        return self.__metrics_df.copy()

    df = property(get_df)

    def get_series_last(self):
        return self.__metrics_last_ser.copy()

    series_last = property(get_series_last)

    @staticmethod
    def create_axes(n_graphs, figsize=None, max_cols=2):
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
        return ax_list

    def plot(self, **kwargs):
        #
        df = self.__metrics_df.set_index(["stage", "epoch"])
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

        ax_list = self.create_axes(n_metrics, **kwargs)

        for i_metric, ax in enumerate(ax_list):
            metric_name = metric_name_list[i_metric]
            metric_df = df.loc[:, metric_name]
            metric_df.columns = metric_df.columns.droplevel(0)
            metric_df.index = metric_df.index.droplevel(0)
            style_list = get_df_column_styles(metric_df)
            metric_df.plot(style=style_list, ms=5, grid=True, ax=ax)
