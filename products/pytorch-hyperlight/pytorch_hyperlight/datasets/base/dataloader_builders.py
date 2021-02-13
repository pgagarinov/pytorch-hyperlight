from abc import ABC, abstractmethod
from functools import partial
from torch.utils.data import DataLoader, Dataset
from pathlib import Path


class ADataLoadersBuilder(ABC):
    DEFAULT_DATASET_ROOT_PATH = None
    DEFAULT_SEED = None
    DEFAULT_DATASET_SUBFOLDER = "_datasets"

    def __init__(self, seed=None, root_path=None, image_size=None):
        if root_path is None:
            root_path = self.DEFAULT_DATASET_ROOT_PATH
        if seed is None:
            seed = self.DEFAULT_SEED
        self.__seed = seed
        self.__root_path = Path(root_path)
        self.__check()

    def __check(self):
        assert self.__root_path is not None
        assert self.__seed is not None

    def build(self):
        f_create_dataloaders = partial(
            self.create_dataloaders,
            seed=self.__seed,
            root_path=self.__root_path,
        )
        f_create_datasets = partial(
            self.create_datasets,
            seed=self.__seed,
            root_path=self.__root_path,
        )
        return f_create_dataloaders, f_create_datasets

    @classmethod
    @abstractmethod
    def create_datasets(cls, *args, **kwargs):
        pass

    @classmethod
    def create_dataloaders(cls, batch_size, n_workers=4, **kwargs):
        #
        SAMPLER = None
        #
        dataset_dict = cls.create_datasets(**kwargs)
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


class AImageDataLoadersBuilder(ADataLoadersBuilder):

    DEFAULT_IMAGE_SIZE = None

    def __init__(self, image_size=None, **kwargs):
        super().__init__(**kwargs)
        if image_size is None:
            image_size = self.DEFAULT_IMAGE_SIZE
        self.__image_size = image_size
        self.__check()

    def get_image_size(self):
        return self.__image_size

    def __check(self):
        assert self.__image_size is not None

    def build(self):
        f_create_dataloaders, f_create_datasets = super().build()

        f_create_dataloaders = partial(
            f_create_dataloaders, image_size=self.__image_size
        )
        f_create_datasets = partial(f_create_datasets, image_size=self.__image_size)
        return f_create_dataloaders, f_create_datasets
