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

import numpy as np


def random_index_split(n_total, val_size, test_size):
    assert val_size >= 0 and val_size < 1
    assert test_size >= 0 and test_size < 1
    assert (test_size + val_size) <= 1
    assert isinstance(n_total, int)
    ind_all_vec = np.arange(n_total)
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
    return ind_train_vec, ind_val_vec, ind_test_vec
