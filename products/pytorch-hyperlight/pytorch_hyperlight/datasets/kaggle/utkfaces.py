# %%
import re
from functools import partial
from pathlib import Path

import albumentations as albu
import numpy as np
import pytorch_lightning as pl
from albumentations.pytorch import ToTensor
from kaggle.api.kaggle_api_extended import KaggleApi
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class UTKFaces(Dataset):
    CLASS_LABELS = ["male", "female"]

    @staticmethod
    def class_inds2labels(class_ind_list):
        label_list = [UTKFaces.CLASS_LABELS[ind] for ind in class_ind_list]
        return label_list

    @staticmethod
    def class_ind2label(class_ind):
        return UTKFaces.CLASS_LABELS[class_ind]

    def get_labels(self):
        return self._labels.copy()

    def __init__(self, file_list, labels, transform=None):
        self._file_list = file_list
        self._labels = np.array(labels, dtype=np.int64)
        self._transform = transform

    def __len__(self):
        return len(self._file_list)

    def __getitem__(self, idx):
        img_path = self._file_list[idx]
        img = Image.open(img_path)
        if self._transform:
            img_transformed = self._transform(image=np.array(img))["image"]
        else:
            img_transformed = img

        id_label = self._labels[idx]
        return img_transformed, id_label


# %%
NORMALIZE_MEAN_LIST = (0.485, 0.456, 0.406)
NORMALIZE_STD_LIST = (0.229, 0.224, 0.225)

# %%
NORMALIZE_T = albu.Normalize(NORMALIZE_MEAN_LIST, NORMALIZE_STD_LIST)

# %%
BORDER_CONSTANT = 0
BORDER_REFLECT = 2


def pre_transforms(image_size):
    # Convert the image to a square of size image_size x image_size
    # (keeping aspect ratio)
    result = [
        albu.LongestMaxSize(max_size=image_size),
        albu.PadIfNeeded(image_size, image_size, border_mode=BORDER_CONSTANT),
    ]

    return result


def hard_transforms():
    result = [
        # Random shifts, stretches and turns with a 50% probability
        albu.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=15,
            border_mode=BORDER_REFLECT,
            p=0.5,
        ),
        albu.IAAPerspective(scale=(0.02, 0.05), p=0.3),
        # Random brightness / contrast with a 30% probability
        albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        # Random gamma changes with a 30% probability
        albu.RandomGamma(gamma_limit=(85, 115), p=0.3),
        # Randomly changes the hue, saturation, and color value of the input image
        albu.HueSaturationValue(p=0.3),
        albu.JpegCompression(quality_lower=80),
    ]

    return result


def show_transforms():
    return [ToTensor()]


def post_transforms():
    # we use ImageNet image normalization
    # and convert it to torch.Tensor
    return [NORMALIZE_T, ToTensor()]


def compose(transforms_to_compose):
    # combine all augmentations into one single pipeline
    result = albu.Compose(
        [item for sublist in transforms_to_compose for item in sublist]
    )
    return result


class UTKFacesDataLoadersBuilder:
    KAGGLE_DATASET_NAME = "jangedoo/utkface-new"
    DATASET_ROOT_SUBFOLDER = "kaggle"
    DEFAULT_DATASET_ROOT_PATH = Path.cwd() / "datasets"
    DEFAULT_SEED = 16
    IMAGE_SIZE = 200

    def __init__(self, seed=None, root_path=None):
        if root_path is None:
            root_path = UTKFacesDataLoadersBuilder.DEFAULT_DATASET_ROOT_PATH
        if seed is None:
            seed = UTKFacesDataLoadersBuilder.DEFAULT_SEED
        self.__seed = seed
        self.__root_path = root_path

    def build(self):
        f_create_dataloaders = partial(
            UTKFacesDataLoadersBuilder.create_dataloaders,
            seed=self.__seed,
            root_path=self.__root_path,
        )
        f_create_datasets = partial(
            UTKFacesDataLoadersBuilder.create_datasets,
            seed=self.__seed,
            root_path=self.__root_path,
        )
        return f_create_dataloaders, f_create_datasets

    @staticmethod
    def create_datasets(
        val_size=0.2, test_size=0.05, root_path=None, label_field="gender", seed=None
    ):
        assert root_path is not None
        assert seed is not None

        api = KaggleApi()
        api.authenticate()
        dataset_dir = (
            root_path
            / UTKFacesDataLoadersBuilder.DATASET_ROOT_SUBFOLDER
            / UTKFacesDataLoadersBuilder.KAGGLE_DATASET_NAME
        )
        if not dataset_dir.exists():
            api.dataset_download_files(
                UTKFacesDataLoadersBuilder.KAGGLE_DATASET_NAME, dataset_dir, unzip=True
            )

        pl.seed_everything(seed)

        all_file_name_list = list((dataset_dir / "UTKFace").glob("*.jpg"))
        age_list, gender_list, etnicity_list = zip(
            *[
                re.split("_", cur_file.name, maxsplit=3)[:3]
                for cur_file in all_file_name_list
            ]
        )

        label_list_dict = {
            "age": age_list,
            "gender": gender_list,
            "etnicity": etnicity_list,
        }

        label_vec = np.array(label_list_dict[label_field])

        image_size = UTKFacesDataLoadersBuilder.IMAGE_SIZE

        TRANSFORM_DICT = {
            "train_augmented": compose(
                [pre_transforms(image_size), hard_transforms(), post_transforms()]
            ),
            "val": compose([pre_transforms(image_size), post_transforms()]),
            "show_augmented": compose(
                [pre_transforms(image_size), hard_transforms(), show_transforms()]
            ),
            "inference": compose([post_transforms()]),
            "show": compose([show_transforms()]),
        }
        #
        all_file_name_vec = np.array(all_file_name_list)

        n_classes = len(np.unique(label_vec))

        ind_all_vec = np.arange(len(label_vec))
        np.random.shuffle(ind_all_vec)

        n_full_train_samples = len(ind_all_vec)
        n_val_samples = int(n_full_train_samples * val_size)
        n_test_samples = int(n_full_train_samples * test_size)
        ind_train_vec, ind_val_vec, ind_test_vec = np.split(
            ind_all_vec,
            np.cumsum(
                [
                    n_full_train_samples - n_val_samples - n_test_samples,
                    n_val_samples,
                ]
            ),
        )

        train_full_augmented_dataset = UTKFaces(
            all_file_name_vec, label_vec, transform=TRANSFORM_DICT["train_augmented"]
        )

        show_train_full_augmented_dataset = UTKFaces(
            all_file_name_vec, label_vec, transform=TRANSFORM_DICT["show_augmented"]
        )

        test_inference_dataset = UTKFaces(
            all_file_name_vec[ind_test_vec],
            label_vec[ind_test_vec],
            transform=TRANSFORM_DICT["inference"],
        )

        test_show_dataset = UTKFaces(
            all_file_name_vec[ind_test_vec],
            label_vec[ind_test_vec],
            transform=TRANSFORM_DICT["show"],
        )

        train_augmented_dataset = UTKFaces(
            all_file_name_vec[ind_train_vec],
            label_vec[ind_train_vec],
            transform=TRANSFORM_DICT["train_augmented"],
        )
        val_dataset = UTKFaces(
            all_file_name_vec[ind_val_vec],
            label_vec[ind_val_vec],
            transform=TRANSFORM_DICT["val"],
        )
        test_dataset = UTKFaces(
            all_file_name_vec[ind_test_vec],
            label_vec[ind_test_vec],
            transform=TRANSFORM_DICT["val"],
        )

        return {
            "train_full_augmented_dataset": train_full_augmented_dataset,
            "train_augmented_dataset": train_augmented_dataset,
            "val_dataset": val_dataset,
            "test_dataset": test_dataset,
            "show_train_full_augmented_dataset": show_train_full_augmented_dataset,
            "test_inference_dataset": test_inference_dataset,
            "test_show_dataset": test_show_dataset,
            "n_classes": n_classes,
        }

    @staticmethod
    def create_dataloaders(batch_size, n_workers=4, **kwargs):
        #
        SAMPLER = None
        #
        dataset_dict = UTKFacesDataLoadersBuilder.create_datasets(**kwargs)
        result_dict = dataset_dict.copy()
        #
        for dataset_name, dataset in dataset_dict.items():
            if isinstance(dataset, Dataset):
                loader_name = dataset_name.replace("dataset", "loader")
                result_dict[loader_name] = DataLoader(
                    dataset_dict[dataset_name],
                    batch_size=batch_size,
                    shuffle=("train" in dataset_name),
                    sampler=SAMPLER,
                    num_workers=n_workers,
                    pin_memory=True,
                )
        return result_dict
