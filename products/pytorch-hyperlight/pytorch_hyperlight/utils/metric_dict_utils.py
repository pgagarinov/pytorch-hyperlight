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


class MetricDictUtils:
    @staticmethod
    def get_prefix_set(metrics_dict):
        return set([k.split("_", 1)[0] for k in metrics_dict.keys()])

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
