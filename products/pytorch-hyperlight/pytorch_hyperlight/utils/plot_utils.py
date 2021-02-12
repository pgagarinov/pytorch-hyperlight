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

import math
import matplotlib.pyplot as plt


def create_subplots(n_graphs, figsize=None, max_cols=2, is_figsize_absolute=False):

    DEFAULT_FIG_SIZE = (20, 12)

    if max_cols is not None:
        n_cols = min(n_graphs, max_cols)
    else:
        n_cols = n_graphs

    n_rows = math.ceil(n_graphs / n_cols)

    if figsize is None:
        figsize = DEFAULT_FIG_SIZE

    if is_figsize_absolute:
        real_figsize = figsize
    else:
        real_figsize = (figsize[0], figsize[1] * n_rows / n_cols)

    fig = plt.figure(figsize=real_figsize)
    ax_list = [None] * n_graphs
    for i_graph in range(n_graphs):
        ax_list[i_graph] = fig.add_subplot(n_rows, n_cols, i_graph + 1)
    return fig, ax_list
