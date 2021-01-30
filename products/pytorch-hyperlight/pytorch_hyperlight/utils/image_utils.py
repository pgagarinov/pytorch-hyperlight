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

import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from pytorch_hyperlight.utils.request_utils import (
    load_url_or_path_as_bytes,
    copy_fileobj_to_s3,
)
import io
from torchvision.utils import save_image
import mimetypes


def _calc_scale_factor(image_size, orig_image_size, fn_opt=max):
    scale_factor = max(
        fn_opt(
            [image_size[0] / orig_image_size[0], image_size[1] / orig_image_size[1]]
        ),
        1,
    )
    return scale_factor


def load_image_as_resized_tensor(image_url_or_path, image_size=None, crop=False):
    image_bytes = load_url_or_path_as_bytes(image_url_or_path)
    image = Image.open(image_bytes)
    image = transforms.ToTensor()(image)
    orig_image_size = list(image.shape)[1:]
    if isinstance(image_size, list):

        scale_factor = _calc_scale_factor(image_size, orig_image_size, fn_opt=max)
        if scale_factor == 1:
            scale_factor = 1 / _calc_scale_factor(
                orig_image_size, image_size, fn_opt=min
            )

        new_image_size = [int(sz * scale_factor) for sz in orig_image_size]
    else:
        new_image_size = image_size
    if new_image_size is not None:
        image = transforms.Resize(new_image_size)(image)
    if crop:
        if image_size is not None:
            image = transforms.CenterCrop(image_size)(image)
    return image


def show_image_tensors(
    image_tensor_list,
    title_list=None,
):
    if not isinstance(image_tensor_list, list):
        image_tensor_list = [image_tensor_list]
    n_images = len(image_tensor_list)
    plt.figure(figsize=(10, 5 * n_images))
    for i_image in range(0, n_images):
        image_tensor = image_tensor_list[i_image]
        plt.subplot(n_images, 1, i_image + 1)
        image_tensor = image_tensor.squeeze()
        image_tensor = image_tensor.permute(1, 2, 0)
        plt.imshow(image_tensor)
        plt.axis("off")
        if title_list is not None:
            title = title_list[i_image]
            plt.title(title)
    plt.show()


def save_image_tensor_to_url(image_tensor, image_url, s3_resource=None):
    if image_url.startswith("s3://"):
        in_mem_file = io.BytesIO()
        guessed_type, _ = mimetypes.guess_type(image_url)
        image_type, guessed_format = guessed_type.split("/")
        assert image_type == "image"
        save_image(image_tensor, in_mem_file, format=guessed_format)
        in_mem_file.seek(0)
        copy_fileobj_to_s3(in_mem_file, image_url, s3_resource=s3_resource)
    else:  # assume local file
        save_image(image_tensor, image_url)
