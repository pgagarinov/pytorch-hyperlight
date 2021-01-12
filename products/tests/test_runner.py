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
import pytest

class TestRunner:
    @pytest.fixture
    def dummy_prerequisites_dict(self):
        import torch
        from pytorch_hyperlight.tasks.classification import ClassificationTaskAdamWWarmup

        # noinspection PyProtectedMember
        from torch.utils.data import DataLoader, TensorDataset
        from torch import nn
        import pytorch_hyperlight as pth

        def create_random_classes_datasets(
                input_shape, n_classes, n_train=1024, n_valid=256, n_test=128
        ):

            train_x = torch.randn([n_train] + input_shape)
            valid_x = torch.randn([n_valid] + input_shape)
            test_x = torch.randn([n_test] + input_shape)

            train_y = torch.randint(n_classes, [n_train], dtype=torch.long)
            valid_y = torch.randint(n_classes, [n_valid], dtype=torch.long)
            test_y = torch.randint(n_classes, [n_test], dtype=torch.long)

            train_dataset = TensorDataset(train_x, train_y)
            valid_dataset = TensorDataset(valid_x, valid_y)
            test_dataset = TensorDataset(test_x, test_y)
            return train_dataset, valid_dataset, test_dataset

        def create_datasets(n_classes=None):
            DEFAULT_N_CLASSES = 10
            INPUT_SHAPE = [1, 28, 28]
            if n_classes is None:
                n_classes = DEFAULT_N_CLASSES

            train_dataset, val_dataset, test_dataset = create_random_classes_datasets(
                INPUT_SHAPE, n_classes
            )

            return {
                "full_train_dataset": train_dataset + val_dataset,
                "train_dataset": train_dataset,
                "val_dataset": val_dataset,
                "test_dataset": test_dataset,
                "n_classes": n_classes,
            }

        # noinspection DuplicatedCode,PyUnusedLocal
        def configure_dataloaders(batch_size, n_workers=4, n_classes=None):
            #
            SHUFFLE = True
            SAMPLER = None
            #
            dataset_dict = create_datasets(n_classes=n_classes)
            loaders_dict = dataset_dict
            #
            for prefix in ["train", "full_train"]:
                loaders_dict[f"{prefix}_loader"] = DataLoader(
                    dataset_dict[f"{prefix}_dataset"],
                    batch_size=batch_size,
                    shuffle=SHUFFLE,
                    sampler=SAMPLER,
                    pin_memory=True,
                )
            #
            for prefix in ["val", "test"]:
                loaders_dict[f"{prefix}_loader"] = DataLoader(
                    dataset_dict[f"{prefix}_dataset"],
                    batch_size=batch_size,
                    shuffle=False,
                    pin_memory=True,
                )
            return loaders_dict

        N_CLASSES = 10
        EXPERIMENT_ID = TestRunner.test_grace_period.__name__
        loaders_dict = configure_dataloaders(32, n_classes=N_CLASSES)

        class DummyModel(nn.Module):
            def __init__(self, n_classes):
                super().__init__()
                self.__n_classes = n_classes
                self.dummy_layer = nn.Linear(1, 1)

            def forward(self, x):
                return torch.ones(x.shape[0], self.__n_classes, requires_grad=True, device=x.device)

        class LitDummyModule(ClassificationTaskAdamWWarmup):
            def __init__(self, hparams):
                n_classes = hparams["n_classes"]
                model = DummyModel(n_classes)
                criterion = nn.CrossEntropyLoss()
                super().__init__(hparams, model, criterion)

        N_CLASSES = loaders_dict["n_classes"]
        IS_CUDA = torch.cuda.is_available()
        GPU_PER_TRIAL = 0.3 * IS_CUDA

        # %%

        CONFIG = {
            "lr": 1e-3,  # Initial learning rate
            "gradient_clip_val": 0,
            "max_epochs": 100,  # the actual number can be less due to early stopping
            "batch_size": 64,
            "n_classes": N_CLASSES,
            "metric_list": ["f1", "acc", "prec"],
            "warmup": 500,
        }

        TUNE_CONFIG = {
            "seed": 16,  # just remove this if you do not want determenistic behavior
            "metric_to_optimize": "val_f1_epoch",  # Ray + PTL Trainer
            "ray_metrics_to_show": [
                "val_loss_epoch",
                "val_f1_epoch",
                "val_acc_epoch",
            ],  # for Ray Tune
            "metric_opt_mode": "max",  # Ray + PTL Trainer
            "cpu_per_trial": 3,  # Ray + DataLoaders
            "gpu_per_trial": GPU_PER_TRIAL,  # for Ray Tune
            "n_checkpoints_to_keep": 1,  # for Ray Tune
            "grace_period": 3,  # for both PTL Trainer and Ray Tune scheduler
            "epoch_upper_limit": 45,  # for Ray Tune
            "n_samples": 3,  # for Ray Tune
            "ptl_early_stopping_patience": 2,  # for PTL Trainer
            "ptl_precision": 32,  # or 16, for PTL Trainer
            "train_loader_name": "train_loader",
            "val_loader_name": "val_loader",
            "test_loader_name": "test_loader",
            "batch_size_main": CONFIG[
                "batch_size"
            ],  # batch size for revalidation and test phases
            #    that run in the main process after all Ray Tune child processes are finished
            "gpus": -1
            * IS_CUDA,  # -1 - use GPU if available, 0 - use CPU, 1 - use single GPU,
            # >=2 - use multiple GPUs
        }
        runner = pth.Runner(
            configure_dataloaders, experiment_id=EXPERIMENT_ID, log2wandb=False
        )

        runner.show_metric_report()
        runner.get_metrics()

        return {
            "runner": runner,
            "config": CONFIG,
            "extra_config": TUNE_CONFIG,
            "lmodule_class": LitDummyModule,
        }

    def test_grace_period(self, dummy_prerequisites_dict):
        runner = dummy_prerequisites_dict["runner"]
        config_dict = dummy_prerequisites_dict["config"]
        extra_config_dict = dummy_prerequisites_dict["extra_config"]
        lmodule_class = dummy_prerequisites_dict["lmodule_class"]

        runner.show_metric_report()
        runner.get_metrics()

        best_result = runner.run_single_trial(
            lmodule_class, config_dict, extra_config_dict
        )
        best_result["metrics"].show_report()

        assert (
            max(best_result["metrics"].df.epoch)
            == extra_config_dict["grace_period"]
            + extra_config_dict["ptl_early_stopping_patience"]
        )

    def test_duplicate_subsequent_runs_same_class_happy_path(
        self, dummy_prerequisites_dict
    ):
        runner = dummy_prerequisites_dict["runner"]
        config_dict = dummy_prerequisites_dict["config"]
        extra_config_dict = dummy_prerequisites_dict["extra_config"]
        lmodule_class = dummy_prerequisites_dict["lmodule_class"]

        runner.show_metric_report()
        runner.get_metrics()

        N_RUNS = 3
        for _ in range(N_RUNS):
            best_result = runner.run_single_trial(
                lmodule_class, config_dict, extra_config_dict
            )
            best_result["metrics"].show_report()

        metrics = runner.get_metrics()
        assert metrics["run_x_last_metric_df"].shape[0] == N_RUNS
