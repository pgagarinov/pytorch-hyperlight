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

    def plot(self):
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

        n_rows = int(math.sqrt(n_metrics))

        n_cols = math.ceil(n_metrics / n_rows)

        fig = plt.figure(figsize=(4 * n_cols, 3 * n_rows))
        for i_metric in range(n_metrics):
            ax = fig.add_subplot(n_rows, n_cols, i_metric + 1)
            metric_name = metric_name_list[i_metric]
            metric_df = df.loc[:, metric_name]
            metric_df.columns = metric_df.columns.droplevel(0)
            metric_df.index = metric_df.index.droplevel(0)
            metric_df.plot(linestyle="-", marker=".", grid=True, ax=ax)
        plt.tight_layout()
