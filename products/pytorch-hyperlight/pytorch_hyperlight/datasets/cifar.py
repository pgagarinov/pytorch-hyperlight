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
#

from pathlib import Path
import torchvision
from pytorch_hyperlight.datasets.base.dataloader_builders import (
    AImageDataLoadersBuilder,
)
from pytorch_hyperlight.utils.random_utils import random_index_split
from torch.utils.data import Subset
from torchvision import transforms
from functools import partial


class CIFARDataLoadersBuilder(AImageDataLoadersBuilder):
    DEFAULT_DATASET_ROOT_PATH = (
        Path.cwd() / AImageDataLoadersBuilder.DEFAULT_DATASET_SUBFOLDER
    )
    DEFAULT_SEED = 16
    DEFAULT_IMAGE_SIZE = 224
    DEFAULT_DATASET_NAME = "cifar10"

    def __init__(self, dataset_name=None, **kwargs):
        if dataset_name is None:
            dataset_name = self.DEFAULT_DATASET_NAME
        self.__dataset_name = dataset_name
        super().__init__(**kwargs)

    @staticmethod
    def post_transforms():
        NORMALIZATION_MEAN_VEC = [0.5, 0.5, 0.5]
        NORMALIZATION_STD_VEC = [0.5, 0.5, 0.5]
        result = [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=NORMALIZATION_MEAN_VEC, std=NORMALIZATION_STD_VEC
            ),
        ]
        return result

    @staticmethod
    def random_transorms(image_size):
        result = [
            transforms.RandomResizedCrop((image_size, image_size), scale=(0.05, 1.0)),
        ]
        return result

    @staticmethod
    def determenistic_transforms(image_size):
        result = [
            transforms.Resize((image_size, image_size)),
        ]
        return result

    @staticmethod
    def show_transforms():
        return [transforms.ToTensor()]

    @classmethod
    def build_transforms_dict(cls, image_size):
        transforms_dict = {
            "train": transforms.Compose(
                cls.random_transorms(image_size) + cls.post_transforms()
            ),
            "test": transforms.Compose(
                cls.determenistic_transforms(image_size) + cls.post_transforms()
            ),
            "show_augmented": transforms.Compose(
                cls.random_transorms(image_size) + cls.show_transforms()
            ),
        }
        return transforms_dict

    @classmethod
    def create_datasets(
        cls,
        val_size=0.2,
        test_size=0.05,
        root_path=None,
        dataset_name="cifar10",
        seed=None,
        image_size=None,
    ):
        assert root_path is not None
        assert seed is not None
        assert image_size is not None
        assert dataset_name in ["cifar100", "cifar10"]

        root_path = Path(root_path) / dataset_name
        n_classes = int(dataset_name.replace("cifar", ""))

        TRANSFORMS_DICT = cls.build_transforms_dict(image_size)
        dataset_class = getattr(torchvision.datasets, dataset_name.upper())

        train_full_augmented_dataset = dataset_class(
            root=root_path,
            train=True,
            download=True,
            transform=TRANSFORMS_DICT["train"],
        )

        train_show_full_augmented_dataset = dataset_class(
            root=root_path,
            train=True,
            download=True,
            transform=TRANSFORMS_DICT["show_augmented"],
        )

        train_full_dataset = dataset_class(
            root=root_path,
            train=True,
            download=True,
            transform=TRANSFORMS_DICT["test"],
        )

        ind_train_vec, ind_val_vec, _ = random_index_split(
            len(train_full_augmented_dataset), val_size, 0
        )

        train_augmented_dataset = Subset(train_full_augmented_dataset, ind_train_vec)
        val_dataset = Subset(train_full_dataset, ind_val_vec)

        test_dataset = dataset_class(
            root=root_path,
            train=False,
            download=True,
            transform=TRANSFORMS_DICT["test"],
        )

        dataset_dict = {
            "train_full_augmented_dataset": train_full_augmented_dataset,
            "train_full_dataset": train_full_dataset,
            "train_augmented_dataset": train_augmented_dataset,
            "train_show_full_augmented_dataset": train_show_full_augmented_dataset,
            "val_dataset": val_dataset,
            "test_dataset": test_dataset,
            "n_classes": n_classes,
        }

        return dataset_dict

    def build(self):
        f_create_dataloaders, f_create_datasets = super().build()

        f_create_dataloaders = partial(
            f_create_dataloaders, dataset_name=self.__dataset_name
        )
        f_create_datasets = partial(f_create_datasets, dataset_name=self.__dataset_name)
        return f_create_dataloaders, f_create_datasets
