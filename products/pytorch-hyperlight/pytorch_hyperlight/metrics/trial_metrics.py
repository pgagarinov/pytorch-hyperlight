import pandas as pd
import math
import matplotlib.pyplot as plt


class TrialMetrics:
    def __init__(self, metrics_last_ser, metrics_df):
        self.__metrics_df = metrics_df
        self.__metrics_last_ser = metrics_last_ser

    def get_metrics(self):
        return self.__metrics_df.copy()

    metrics = property(get_metrics)

    def get_metrics_last(self):
        return self.__metrics_last_ser.copy()

    metrics_last = property(get_metrics_last)

    @staticmethod
    def create_axes(n_graphs, figsize=None):
        SUBPLOT_WIDTH = 12
        SUBPLOT_HEIGHT = 7
        MAX_COLS = 2
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
            metric_df.plot(linestyle="-", marker=".", grid=True, ax=ax)
        plt.tight_layout()
