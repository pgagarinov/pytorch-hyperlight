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

import torch
import pandas as pd
from copy import deepcopy
import re


class MetricDictUtils:
    @staticmethod
    def get_stage_name_pretty(stage_list):
        # stage can be 'train', 'val' and 'test'
        return "/".join([s.capitalize() for s in stage_list])

    @staticmethod
    def filter_n_round_epoch_metrics(metrics_dict):
        N_DIGITS_AFTER_DOT = 4

        metrics_dict = MetricDictUtils.filter_by_suffix(metrics_dict, "_epoch")
        metrics_dict = MetricDictUtils.remove_suffix(metrics_dict, "_epoch")
        metrics_dict = MetricDictUtils.round_floats(metrics_dict, N_DIGITS_AFTER_DOT)
        return metrics_dict

    @staticmethod
    def metrics_dict2df(metrics_dict, epoch, stage_list):
        stage_name_pretty = MetricDictUtils.get_stage_name_pretty(stage_list)
        metrics_dict = deepcopy(metrics_dict)
        metrics_dict["stage-list"] = [stage_list]
        metrics_dict["stage"] = stage_name_pretty

        # sort metric names in alphabetical order from the tail
        sorted_metric_list = [
            e[::-1] for e in sorted([e[::-1] for e in list(metrics_dict.keys())])
        ]
        #
        metrics_df = pd.DataFrame(
            metrics_dict, index=[epoch], columns=sorted_metric_list
        )
        metrics_df.index.name = "epoch"
        #
        return metrics_df

    @staticmethod
    def metrics_df_list_concat(metrics_df_list):
        epoch_list = [e.index[0] for e in metrics_df_list]
        df = pd.concat(metrics_df_list)
        df.insert(loc=0, column="epoch", value=epoch_list)
        df.reset_index(inplace=True, drop=True)
        return df

    @staticmethod
    def get_list_prefix_list(key_list, split_symbol_list=("_")):
        if isinstance(split_symbol_list, str):
            split_symbol_list = split_symbol_list

        split_re = "|".join(split_symbol_list)
        return [re.split(split_re, k, 1)[0] for k in key_list]

    @staticmethod
    def get_prefix_list(metrics_dict):
        return MetricDictUtils.get_list_prefix_list(metrics_dict.keys())

    @staticmethod
    def filter_by_prefix(metrics_dict, prefixes2keep_list):
        key_list = metrics_dict.keys()
        prefix_list = MetricDictUtils.get_list_prefix_list(key_list)
        res_metrics_dict = {
            key: metrics_dict[key]
            for prefix, key in zip(prefix_list, key_list)
            if prefix in prefixes2keep_list
        }
        return res_metrics_dict

    @staticmethod
    def get_prefix_set(metrics_dict):
        return set(MetricDictUtils.get_prefix_list(metrics_dict))

    @staticmethod
    def strip_tensors(metrics_dict):
        return {
            k: v.cpu().item()
            for k, v in metrics_dict.items()
            if isinstance(v, torch.Tensor)
        }

    @staticmethod
    def filter_by_suffix(metrics_dict, suffix):
        res_metric_dict = {k: v for k, v in metrics_dict.items() if k.endswith(suffix)}
        return res_metric_dict

    @staticmethod
    def remove_suffix(metrics_dict, suffix):
        res_metric_dict = {k.replace(suffix, ""): v for k, v in metrics_dict.items()}
        return res_metric_dict

    @staticmethod
    def change_prefix(metrics_dict, from_prefix, to_prefix):
        res_metric_dict = {
            k.replace(from_prefix, to_prefix) if k.startswith(from_prefix) else k: v
            for k, v in metrics_dict.items()
        }
        return res_metric_dict

    @staticmethod
    def round_floats(metrics_dict, n_digits_after_dot):
        metrics_dict = {
            k: round(v, n_digits_after_dot) if isinstance(v, float) else v
            for k, v in metrics_dict.items()
        }
        return metrics_dict
