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

import io

import numpy as np
from PIL import Image
from pytorch_hyperlight.tasks.nst import NSTImageUrlOrPathDataLoader


def generate_image_inmem_file(size_vec):
    arr = np.random.randint(0, 255, list(size_vec) + [3], dtype="uint8")
    img = Image.fromarray(arr)
    in_mem_file = io.BytesIO()
    img.save(in_mem_file, format="PNG")
    in_mem_file.seek(0)
    return in_mem_file


def rand_size():
    return np.random.randint(100, 1000)


def check(content_image_size, style_image_size_list):
    print(content_image_size)
    print(style_image_size_list)

    content_image_inmem_file = generate_image_inmem_file(content_image_size)
    style_image_inmem_fie_list = [
        generate_image_inmem_file(size_vec) for size_vec in style_image_size_list
    ]

    loader = NSTImageUrlOrPathDataLoader(
        10, 600, content_image_inmem_file, style_image_inmem_fie_list
    )

    batch = next(iter(loader))
    context_size_vec = batch[0].size()
    for style_tensor in batch[1]:
        style_size_vec = style_tensor.size()
        assert style_size_vec == context_size_vec, (
            f"test failed for content_image_size={content_image_size},"
            + f" style_image_inmem_fie_list={style_image_inmem_fie_list}",
        )


class TestRunner:
    def test_image_or_path_dataloader_tricky_sizes(self):
        content_image_size = (318, 511)
        style_image_size_list = [(398, 229), (527, 445)]
        check(content_image_size, style_image_size_list)

    def test_image_or_path_dataloader_random_sizes(self):
        for _ in range(10):
            content_image_size = (rand_size(), rand_size())
            style_image_size_list = [
                (rand_size(), rand_size()),
                (rand_size(), rand_size()),
            ]
            check(content_image_size, style_image_size_list)
