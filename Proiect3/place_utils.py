from parameters import Parameters
from scipy import ndarray
from typing import List
import random
import functools
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
from util import get_column_path_dynamicprogramming, transpose


def start_with_params(params: Parameters, i: int, j: int):
    """ Returns the starting coordinates of the (i, j) position.
    Takes into consideration overlap.
    """

    overlap_width = int(params.overlap * params.block_width)
    overlap_height = int(params.overlap * params.block_height)

    (start_height, start_width) = (i * params.block_height, j * params.block_width)
    start_height -= i * overlap_height
    start_width -= j * overlap_width
    return (start_height, start_width)
