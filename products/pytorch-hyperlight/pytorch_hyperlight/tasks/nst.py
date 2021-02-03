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

import enum
import math
import pytorch_lightning as pl
import torch.nn as nn
from abc import ABC, abstractmethod
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import copy
from functools import partial
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.optim import Adam
from pytorch_hyperlight.utils.image_utils import load_image_as_resized_tensor


IMAGE_MEAN_VEC = torch.tensor([0.485, 0.456, 0.406])
IMAGE_STD_VEC = torch.tensor([0.229, 0.224, 0.225])


class ELossLayerMode(enum.Enum):
    DoNothing = 1
    CalcLoss = 2
    CaptureTarget = 3


class ABaseLoss(nn.Module, ABC):
    __slots__ = "__mode", "__ind_batch_item2capture_list", "__target"

    def __init__(self, ind_batch_item2capture_list):
        super().__init__()
        self.__mode = ELossLayerMode.DoNothing
        self.__ind_batch_item2capture_list = ind_batch_item2capture_list

    def set_mode(self, mode: ELossLayerMode):
        self.__mode = mode

    @abstractmethod
    def _calc_loss(self, input, target):
        pass

    def _extract_target(self, input):
        ind2capture_list = self.__ind_batch_item2capture_list
        target = input[ind2capture_list, :, :, :].detach()
        return target

    def forward(self, input):
        assert isinstance(self.__mode, ELossLayerMode)
        if self.__mode == ELossLayerMode.CaptureTarget:
            assert not hasattr(
                self, "_ABaseLoss__target"
            ), "target is not supposed to be captured twice"
            self.__target = nn.Parameter(
                self._extract_target(input), requires_grad=False
            )
        elif self.__mode == ELossLayerMode.CalcLoss:
            self.loss = self._calc_loss(input, self.__target)
        return input


class ContentLoss(ABaseLoss):
    def _calc_loss(self, input, target):
        assert input.shape[0] == 1
        assert target.shape[0] == 1
        loss = F.mse_loss(input, target)
        return loss


class StyleWeigtedLoss(ABaseLoss):
    def __init__(self, ind_batch_item2capture_list, style_weight_list):
        super().__init__(ind_batch_item2capture_list)
        self.__style_weight_vec = torch.tensor(style_weight_list)

    @staticmethod
    def gram_matrix_batched(input):
        a, b, c, d = input.size()
        features = input.view(a, b, c * d)
        gram_batched = torch.matmul(features, features.permute(0, 2, 1))
        return gram_batched.div(b * c * d)

    def _extract_target(self, input):
        target = super()._extract_target(input)
        target_gram = self.gram_matrix_batched(target)
        return target_gram

    def _get_style_gram_matrix4input_list(self, input, n_styles):
        assert input.shape[0] == 1
        gram_matrix_input = self.gram_matrix_batched(input)
        return [gram_matrix_input] * n_styles

    def _calc_loss(self, input, target):
        assert input.shape[0] == 1
        n_styles = len(self.__style_weight_vec)
        assert target.shape[0] == n_styles
        loss = 0
        style_gram_matrix4input_list = self._get_style_gram_matrix4input_list(
            input, n_styles
        )
        for i_style, style_weight in enumerate(self.__style_weight_vec):
            style_gram_matrix4input = style_gram_matrix4input_list[i_style][0]
            loss += F.mse_loss(style_gram_matrix4input, target[i_style]) * style_weight
        return loss


class StylePlainWeigtedMergeLoss(StyleWeigtedLoss):
    pass


class StyleVerticalSplitWeigtedMergeLoss(StyleWeigtedLoss):
    def _get_style_gram_matrix4input_list(self, input, n_styles):
        assert input.shape[0] == 1
        n_width_pixels = input.shape[-1]
        pixel_chunk_size = math.ceil(n_width_pixels / n_styles)
        split_index_vec_list = torch.arange(n_width_pixels).split(pixel_chunk_size)
        style_gram_list = []
        for i_style in range(n_styles):
            cur_style_target = input[0:1, :, :, split_index_vec_list[i_style]]
            cur_target_gram = self.gram_matrix_batched(cur_style_target)
            style_gram_list.append(cur_target_gram)

        return style_gram_list


class LossModel(nn.Module):
    def __init__(
        self,
        model,
        style_weight_list,
        style_loss_list,
        content_weight,
        content_loss_list,
    ):
        super().__init__()
        self.__style_weight_list = style_weight_list
        self.__style_loss_list = nn.ModuleList(style_loss_list)
        self.__content_weight = content_weight
        self.__content_loss_list = nn.ModuleList(content_loss_list)
        self.__model = model

    def __set_mode(self, mode):
        for content_loss in self.__content_loss_list:
            content_loss.set_mode(mode)
        for style_loss in self.__style_loss_list:
            style_loss.set_mode(mode)

    def set_images(self, content_image, style_image_list):
        n_style_images = len(style_image_list)
        n_style_wights = len(self.__style_weight_list)
        assert n_style_images == n_style_wights, \
            (f'number of style images {n_style_images} is different' +
             f' from the number of style weights {n_style_wights}')
        self.__set_mode(ELossLayerMode.CaptureTarget)
        input = torch.cat([content_image] + style_image_list, axis=0)
        self.__model(input)
        self.__set_mode(ELossLayerMode.CalcLoss)

    def forward(self, varying_image):
        self.__model(varying_image)
        style_loss = 0
        for style_loss_layer in self.__style_loss_list:
            style_loss += style_loss_layer.loss

        content_loss = 0
        for content_loss_layer in self.__content_loss_list:
            content_loss += content_loss_layer.loss

        content_loss *= self.__content_weight

        total_loss = style_loss + content_loss
        return total_loss, style_loss, content_loss


# We will use PyTorch-Lightning to concentrate on the model, not on the surrounding code
# The images are passed to the constructor of PyTorch-Lightning module wich makes it
# unnecessary to pass any data in batches.


class StyleLossBuilder:
    @staticmethod
    def build(style_merge_method, style_weight_list):
        if style_merge_method == "plain_merge":
            style_loss_class = StylePlainWeigtedMergeLoss
        elif style_merge_method == "vertical_split_merge":
            style_loss_class = StyleVerticalSplitWeigtedMergeLoss
        else:
            assert False, f"Unknown style_merge_method: {style_merge_method}"
        ind_style_batch_item2capture_list = [
            i_elem + 1 for i_elem, _ in enumerate(style_weight_list)
        ]

        return style_loss_class(ind_style_batch_item2capture_list, style_weight_list)


class ContentLossBuilder:
    @staticmethod
    def build():
        ind_content_batch_item2capture_list = [0]
        return ContentLoss(ind_content_batch_item2capture_list)


class LossModelBuilder:
    @staticmethod
    def generate_layers_n_hook_list(f_create_layer, ind_layer_list):
        hook_list = []
        layer_list = []
        for ind_layer in ind_layer_list:
            cur_layer = f_create_layer()
            hook = partial(
                lambda layer, module, input, output: layer.forward(output), cur_layer
            )
            hook_list.append(hook)
            layer_list.append(cur_layer)

        return layer_list, hook_list

    @staticmethod
    def build(
        orig_model,
        content_layer_name_list,
        style_layer_name_list,
        level_name2id_map_dict,
        content_weight,
        style_weight_list,
        style_merge_method,
    ):

        ind_layer_content_list = [
            level_name2id_map_dict[level] for level in content_layer_name_list
        ]
        ind_layer_style_list = [
            level_name2id_map_dict[level] for level in style_layer_name_list
        ]

        orig_model = copy.deepcopy(orig_model)

        (
            content_loss_list,
            content_hook_list,
        ) = LossModelBuilder.generate_layers_n_hook_list(
            ContentLossBuilder.build, ind_layer_content_list
        )
        style_loss_list, style_hook_list = LossModelBuilder.generate_layers_n_hook_list(
            partial(StyleLossBuilder.build, style_merge_method, style_weight_list),
            ind_layer_style_list,
        )

        for param in orig_model.parameters():
            param.requires_grad = False

        ind_layer_full_list = ind_layer_content_list + ind_layer_style_list
        for ind_layer, hook in zip(
            ind_layer_full_list,
            content_hook_list + style_hook_list,
        ):
            orig_model[ind_layer].register_forward_hook(hook)

        orig_model = orig_model[: (max(ind_layer_full_list) + 1)]

        for layer in orig_model.children():
            if isinstance(layer, nn.ReLU):
                layer.inplace = False

        model = nn.Sequential(
            transforms.Normalize(IMAGE_MEAN_VEC, IMAGE_STD_VEC), orig_model
        )
        loss_model = LossModel(
            model,
            style_weight_list,
            style_loss_list,
            content_weight,
            content_loss_list,
        )

        return loss_model


class LossVGG19ModelBuilder:
    LEVEL_MAPPING_VGG19_DICT = {
        "conv1_1": 0,
        "conv1_2": 2,
        "conv2_1": 5,
        "conv2_2": 7,
        "conv3_1": 10,
        "conv3_2": 12,
        "conv3_3": 14,
        "conv3_4": 16,
        "conv4_1": 19,
        "conv4_2": 21,
        "conv4_3": 23,
        "conv4_4": 25,
        "conv5_1": 28,
        "conv5_2": 30,
        "conv5_3": 32,
        "conv5_4": 34,
    }

    def build(hparams):
        cnn = models.vgg19(pretrained=True, progress=True).features
        style_weight_list = hparams["style_weight_list"]
        content_weight = hparams["content_weight"]
        loss_model = LossModelBuilder.build(
            cnn,
            hparams["content_layer_name_list"],
            hparams["style_layer_name_list"],
            LossVGG19ModelBuilder.LEVEL_MAPPING_VGG19_DICT,
            content_weight,
            style_weight_list,
            hparams["style_merge_method"],
        )
        return loss_model


class LitVGG19StyleTransfer(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        #
        loss_model = LossVGG19ModelBuilder.build(hparams)
        #
        self.loss_model = loss_model
        self.__is_image_set = False
        self.varying_image = nn.Parameter(
            torch.randn([1] + self.hparams.varying_image_tensor_size_list)
        )
        assert len(self.varying_image.shape) == 4

    def configure_optimizers(self):

        optimizer = Adam([self.varying_image], lr=self.hparams.lr)

        return optimizer

    def __calc_n_log_loss(self):

        total_loss, style_loss, content_loss = self.loss_model.forward(
            self.varying_image
        )
        self.log(
            "train_style_loss", style_loss, prog_bar=True, on_step=True, on_epoch=True
        )
        self.log(
            "train_content_loss",
            content_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "train_total_loss", total_loss, prog_bar=True, on_step=True, on_epoch=True
        )
        return total_loss

    # we ignore batch data that comes into this method
    def training_step(self, batch, __):
        if self.__is_image_set is False:
            content_image = batch[0]
            style_image_list = batch[1]
            assert content_image is not None
            assert style_image_list is not None
            self.loss_model.set_images(content_image, style_image_list)
            self.varying_image.data = content_image.data.clone()
            self.__is_image_set = True

        loss = self.__calc_n_log_loss()
        return {"loss": loss}

    def optimizer_step(
        self,
        current_epoch,
        batch_nb,
        optimizer,
        optimizer_idx,
        closure,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        optimizer.step(closure=closure)
        self.varying_image.data.clamp_(0, 1)


# Since PyTorch-Lightning assumes is passed via dataloader we will create the dummy dataloader
# with the single purpose of triggering the PyTorch-Lightning optimization loop. We will also assume
# that our batch size is 1, number of steps per epoch is defined by n_dummy_samples parameter.
# This way we will minimize PyTorch-Lightning overhead (up to ~300ms per epoch) which would hit us
# if we used 1 step per epoch.


class NSTImageTensorDataset(Dataset):
    def __init__(self, n_dummy_samples, content_image_tensor, style_image_tensor_list):
        self.__n_dummy_samples = n_dummy_samples
        self.__content_image_tensor = content_image_tensor
        self.__style_image_tensor_list = style_image_tensor_list

    def __getitem__(self, ind):
        if ind == 0:
            x = self.__content_image_tensor
            y = self.__style_image_tensor_list
        else:  # dummy data
            x = -1
            y = -1
        return x, y

    def __len__(self):
        return self.__n_dummy_samples


class NSTImageTensorDataLoader(DataLoader):
    #
    #  dummy samples serve the only purpose of reducing the epoch overhead of PyTorch-Lightning
    #  without dummy samples would only have a single sample in the dataset which would lead to
    #  incrementing the epoch number at each SGD step. Epoch increment causes an overhead of up to
    #  300ms in PyTorch Lightning. When N_DUMMY_SAMPLES> 1 and batch_size = 1 the epoch increment
    #  happens every N_DUMMY_SAMPLES samples
    #
    def __init__(self, n_dummy_samples, content_image, style_image_list):
        train_dataset = NSTImageTensorDataset(
            n_dummy_samples, content_image, style_image_list
        )
        super().__init__(train_dataset)


class NSTImageUrlOrPathDataLoader(NSTImageTensorDataLoader):
    def __init__(
        self, n_dummy_samples, target_image_height, content_file, style_file_list
    ):
        content_image_tensor = load_image_as_resized_tensor(
            content_file, image_size=target_image_height
        )
        style_image_tensor_list = []
        for style_file in style_file_list:
            style_image_tensor = load_image_as_resized_tensor(
                style_file,
                image_size=list(content_image_tensor.shape)[1:],
                crop=True,
            )
            style_image_tensor_list.append(style_image_tensor)

        super().__init__(n_dummy_samples, content_image_tensor, style_image_tensor_list)
