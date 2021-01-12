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

from pytorch_hyperlight.runner.base_runner import BaseRunner
from pytorch_hyperlight.metrics.inmem_metric_storage import InMemMetricStorage
from collections import Counter


class Runner(BaseRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__metric_storage = InMemMetricStorage()
        self.__run_name_duplicate_counter = Counter()

    def __log_metrics(self, result_dict, run_name):
        self.__metric_storage.log_trial_metrics(result_dict["metrics"], run_name)

    def __generate_run_name(
        self, run_name, run_name_prefix, extra_run_name_prefix, lmodule_class
    ):
        assert run_name is None or (
            run_name_prefix is None and extra_run_name_prefix is None
        ), "Cannot combine run_name with prefixes"
        if run_name is None:
            run_name = extra_run_name_prefix + run_name_prefix + lmodule_class.__name__

            if (extra_run_name_prefix == "") and (
                run_name in self.__run_name_duplicate_counter
            ):
                cnt = self.__run_name_duplicate_counter[run_name]
                cnt += 1
                self.__run_name_duplicate_counter[run_name] = cnt
                run_name = f"{run_name}@{cnt}"
            else:
                self.__run_name_duplicate_counter[run_name] = 1

        return run_name

    def run_single_trial(
        self,
        lmodule_class,
        *args,
        run_name=None,
        run_name_prefix="single-trial-",
        extra_run_name_prefix="",
        **kwargs,
    ):
        result_dict = super().run_single_trial(lmodule_class, *args, **kwargs)
        run_name = self.__generate_run_name(
            run_name, run_name_prefix, extra_run_name_prefix, lmodule_class
        )
        self.__log_metrics(result_dict, run_name)
        return result_dict

    def run_hyper_opt(
        self,
        lmodule_class,
        *args,
        run_name=None,
        run_name_prefix="hyper-opt-",
        extra_run_name_prefix="",
        **kwargs,
    ):
        result_dict = super().run_hyper_opt(lmodule_class, *args, **kwargs)
        run_name = self.__generate_run_name(
            run_name, run_name_prefix, extra_run_name_prefix, lmodule_class
        )
        self.__log_metrics(result_dict, run_name)
        return result_dict

    def show_metric_report(self, **kwargs):
        self.__metric_storage.show_report(**kwargs)

    def get_metrics(self, **kwargs):
        return self.__metric_storage.get_metrics(**kwargs)
