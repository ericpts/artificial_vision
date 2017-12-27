from parameters import Parameters
from scipy import ndarray
from typing import List
import random
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
from util import get_column_path_dynamicprogramming, transpose, image_energy, to_grayscale


def generate_blocks(nblocks: int, block_height: int, block_width: int, sample: ndarray) -> List:
    blocks = []
    (sample_height, sample_width) = sample.shape[0:2]

    max_cnt = sample_height * sample_width

    if nblocks > 4 * max_cnt:
        nblocks = 4 * max_cnt

    for _ in range(nblocks):
        h = random.randint(0, sample_height - block_height)
        w = random.randint(0, sample_width - block_width)

        block = sample[h:h + block_height, w:w + block_width]
        blocks.append(block)

    for b in blocks:
        assert b.shape[0:2] == (block_height, block_width)

    return blocks


def distance_matrix(x: ndarray, y: ndarray) -> float:
    x = x.astype(float)
    y = y.astype(float)

    m = (x - y)**2
    if len(m.shape) == 3:
        return np.sum(m, axis=(2,))
    else:
        return m


def distance(x: ndarray, y: ndarray) -> float:
    return np.sum(distance_matrix(x, y))


def start_with_params(params: Parameters, i: int, j: int):
    """ Returns the starting coordinates of the (i, j) position.
    Takes into consideration overlap.
    """
    (start_height, start_width) = (i * params.block_height, j * params.block_width)
    start_height -= i * params.overlap_height
    start_width -= j * params.overlap_width
    return (start_height, start_width)
