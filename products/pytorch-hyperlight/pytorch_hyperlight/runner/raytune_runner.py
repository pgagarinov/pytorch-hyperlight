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

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch


def tune_init():
    ray.shutdown()
    ray.init(dashboard_host="0.0.0.0", log_to_driver=False)


def run_tune_experiment_asha_hyperopt(
    trainer_func, search_space_config, tune_config, raytune_loggers=None
):
    """
    This is high-level function that launches an automatic hyper-parameter search via Ray Tune.
    The function expects the trainer function, the hyper-parameter space and the technical parameters.
    """
    if raytune_loggers is None:
        raytune_loggers = []

    exp_config = search_space_config.copy()

    asha_kwargs = {k: v for k, v in tune_config.items() if k == "grace_period"}
    #
    asha_scheduler = ASHAScheduler(
        time_attr="training_iteration",
        max_t=tune_config["epoch_upper_limit"],
        reduction_factor=3,
        brackets=1,
        **asha_kwargs
    )

    reporter = CLIReporter(
        parameter_columns=list(exp_config.keys()),
        metric_columns=tune_config["ray_metrics_to_show"] + ["training_iteration"],
        max_progress_rows=tune_config["n_samples"],
    )
    #

    # HyperOpt algorithm selects hyper-parameters for next trials
    #  while ASHAScheduler eliminates not-promising trials early
    #  freeing computational resources for more promising ones
    hyperopt_search = HyperOptSearch()

    ray_tune_loggers = list(DEFAULT_LOGGERS) + raytune_loggers

    analysis = tune.run(
        trainer_func,
        scheduler=asha_scheduler,
        search_alg=hyperopt_search,
        resources_per_trial={
            "cpu": tune_config["cpu_per_trial"],
            "gpu": tune_config["gpu_per_trial"],
        },
        metric=tune_config["metric_to_optimize"],
        mode=tune_config["metric_opt_mode"],
        config=exp_config,
        num_samples=tune_config["n_samples"],
        progress_reporter=reporter,
        checkpoint_score_attr=tune_config["metric_to_optimize"],
        keep_checkpoints_num=tune_config["n_checkpoints_to_keep"],
        name=tune_config["experiment_id"],
        verbose=1,
        loggers=ray_tune_loggers,
    )

    print("Best hyperparameters found were: ", analysis.best_config)

    return analysis
