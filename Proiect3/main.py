#!/usr/bin/python3

import pdb
import sys
import time
import argparse
import random
import itertools
import functools
import matplotlib.pyplot as plt

import skimage.color
import skimage.filters
from skimage.viewer import ImageViewer
from skimage.viewer.canvastools import RectangleTool

from typing import List

import numpy as np

import scipy.spatial.distance
from scipy import ndarray
from scipy.ndimage import convolve

from tqdm import tqdm

sys.path.append('../')
from util import *

def distance(x: ndarray, y: ndarray) -> float:
    return np.linalg.norm(x - y)

INF = 2**60

class Parameters(object):
    def __init__(self,
            output_width: int, output_height: int,
            block_width: int, block_height: int,
            blocks_per_width: int, blocks_per_height: int,
            nchannels: int,
            overlap: float = None):

        self.output_width = output_width
        self.output_height = output_height
        self.block_width = block_width
        self.block_height = block_height
        self.blocks_per_width = blocks_per_width
        self.blocks_per_height = blocks_per_height

        self.nchannels = nchannels
        self.overlap = overlap


def make_output_no_overlap(params: Parameters, indexes: ndarray, blocks: List):
    """ Generates the output image from the given indexes and blocks.
        Does not do anything with overlap.
    """
    output = ndarray(shape=(params.output_width, params.output_height, params.nchannels), dtype=np.uint8)
    for i in range(params.blocks_per_width):
        for j in range(params.blocks_per_height):
            (start_width, start_height) = (i * params.block_width, j * params.block_height)
            (end_width, end_height) = (start_width + params.block_width, start_height + params.block_height)

            output[start_width: end_width, start_height: end_height] = blocks[indexes[i][j]]
    return output

def place_random(params: Parameters, blocks: List) -> ndarray:
    """ Returns the generated picture. """
    output_indexes = ndarray(shape=(params.blocks_per_width, params.blocks_per_height), dtype=int)
    output_indexes[0, 0] = random.randint(0, len(blocks))

    def get_best_fit(left, up):
        return random.randint(0, len(blocks))

    for i in tqdm(range(params.blocks_per_width)):
        for j in tqdm(range(params.blocks_per_height)):
            if i == 0 and j == 0:
                # We have already placed the corner.
                continue

            left = output_indexes[i][j - 1] if j > 0 else None
            up = output_indexes[i - 1][j] if i > 0 else None

            output_indexes[i][j] = get_best_fit(left, up)
    return make_output_no_overlap(params, output_indexes, blocks)

def make_output_with_overlap(params: Parameters, indexes: ndarray, blocks: List):
    """ Generates the output image from the given indexes and blocks.
        Considers overlaps.
    """
    overlap_width = int(params.block_width * params.overlap)
    overlap_height = int(params.block_height * params.overlap)

    output = ndarray(shape=(params.output_width - overlap_width, params.output_height - overlap_height, params.nchannels), dtype=np.uint8)

    for i in range(params.blocks_per_width):
        for j in range(params.blocks_per_height):
            (start_width, start_height) = (i * params.block_width, j * params.block_height)

            if i > 0:
                start_width -= overlap_width
            if j > 0:
                start_height -= overlap_height

            (end_width, end_height) = (start_width + params.block_width, start_height + params.block_height)

            output[start_width: end_width, start_height: end_height] = blocks[indexes[i][j]]
    return output

def place_overlap(params: Parameters, blocks: List) -> ndarray:
    """ Returns the generated picture. """
    output_indexes = ndarray(shape=(params.blocks_per_width, params.blocks_per_height), dtype=int)
    output_indexes[0, 0] = random.randint(0, len(blocks))

    @functools.lru_cache()
    def distance_horizontal(left_i: int, right_i: int) -> float:
        """ Calculates the overlap cost of placing `left` and `right` together. """
        left = blocks[left_i]
        right = blocks[right_i]
        delta_width = int(params.overlap * params.block_width)
        return distance(left[:, -delta_width: ], right[:, : delta_width])

    @functools.lru_cache()
    def distance_vertical(up_i: int, down_i: int) -> float:
        """ Calculates the overlap cost of placing `up` and `down` together. """
        up = blocks[up_i]
        down = blocks[down_i]
        delta_height = int(params.overlap * params.block_height)
        return distance(up[-delta_height: ,:], down[delta_height, :])

    def get_best_fit(left: int, up: int) -> int:
        """ Get the best index given the two neighbours. """
        def cost(i):
            ret = 0
            if left:
                ret += distance_horizontal(left, i)
            if up:
                ret += distance_vertical(up, i)
            return ret
        best, best_cost = (0, cost(0))

        sample_blocks = random.sample(range(len(blocks)), min(1000, len(blocks)))
# Only choose a subset of blocks to consider.
# If we do not do this, the picture will end up consisting of the same 3-4 blocks repeated over and over.
        for i in sample_blocks:
            now_cost = cost(i)
            if now_cost < best_cost:
                best = i
        return best

    for i in tqdm(range(params.blocks_per_width)):
        for j in range(params.blocks_per_height):
            if i == 0 and j == 0:
                # We have already placed the corner.
                continue

            left = output_indexes[i][j - 1] if j > 0 else None
            up = output_indexes[i - 1][j] if i > 0 else None

            output_indexes[i][j] = get_best_fit(left, up)
    return make_output_with_overlap(params, output_indexes, blocks)

def make_output_with_overlap_and_edge_cut(params: Parameters, indexes: ndarray, blocks: List):
    """ Generates the output image from the given indexes and blocks.
        Considers overlaps and also appropriately cuts edges.
    """
    overlap_width = int(params.block_width * params.overlap)
    overlap_height = int(params.block_height * params.overlap)

    output = ndarray(shape=(params.output_width - overlap_width, params.output_height - overlap_height, params.nchannels), dtype=np.uint8)

    for i in range(params.blocks_per_width):
        for j in range(params.blocks_per_height):
            (start_width, start_height) = (i * params.block_width, j * params.block_height)

            if i > 0:
                start_width -= overlap_width
            if j > 0:
                start_height -= overlap_height

            (end_width, end_height) = (start_width + params.block_width, start_height + params.block_height)

            output[start_width: end_width, start_height: end_height] = blocks[indexes[i][j]]
    return output

def place_overlap_and_edge_cut(params: Parameters, blocks: List) -> ndarray:
    """ Returns the generated picture. """
    output_indexes = ndarray(shape=(params.blocks_per_width, params.blocks_per_height), dtype=int)
    output_indexes[0, 0] = random.randint(0, len(blocks))

    def energy_horizontal(left_i: int, right_i: int):
        left = blocks[left_i]
        right = blocks[right_i]
        delta_width = int(params.overlap * params.block_width)

        left_side = left[:, -delta_width: ]
        right_side = right[:, : delta_width]

        energy = np.sum(
                (left_side - right_side) ** 2,
                axis=(2))
        return energy

    def distance_horizontal(left_i: int, right_i: int):
        """ Calculates the overlap cost of placing `left` and `right` together. """

        energy = energy_horizontal(left_i, right_i)
        path = get_column_path_dynamicprogramming(energy)

        ret = 0
        for (i, j) in path.items():
            ret += energy[i, j]
        return ret

    def energy_vertical(up_i: int, down_i: int):
        """ Calculates the overlap cost of placing `up` and `down` together. """
        up = blocks[up_i]
        down = blocks[down_i]
        delta_height = int(params.overlap * params.block_height)

        up_side = up[-delta_height: ,:]
        down_side = down[delta_height, :]

        energy = np.sum(
                (up_side - down_side) ** 2,
                axis=(2))
        return energy

    def distance_vertical(up_i: int, down_i: int):
        """ Calculates the overlap cost of placing `up` and `down` together. """

        energy = energy_vertical(up_i, down_i)
        path = get_column_path_dynamicprogramming(transpose(energy))

        ret = 0
        for (i, j) in path.items():
            ret += energy[j, i]
        return ret

    def get_best_fit(left: int, up: int) -> int:
        """ Get the best index given the two neighbours. """
        def cost(i):
            ret = 0
            if left:
                ret += distance_horizontal(left, i)
            if up:
                ret += distance_vertical(up, i)
            return ret
        best, best_cost = (0, cost(0))

        sample_blocks = random.sample(range(len(blocks)), min(1000, len(blocks)))
# Only choose a subset of blocks to consider.
# If we do not do this, the picture will end up consisting of the same 3-4 blocks repeated over and over.
        for i in sample_blocks:
            now_cost = cost(i)
            if now_cost < best_cost:
                best = i
        return best

    for i in tqdm(range(params.blocks_per_width)):
        for j in range(params.blocks_per_height):
            if i == 0 and j == 0:
                # We have already placed the corner.
                continue

            left = output_indexes[i][j - 1] if j > 0 else None
            up = output_indexes[i - 1][j] if i > 0 else None

            output_indexes[i][j] = get_best_fit(left, up)


    return make_output_with_overlap_and_edge_cut(params, output_indexes, blocks)

def main():

    random.seed(1337)

    parser = argparse.ArgumentParser(description = 'Texture generator.')
    parser.add_argument('--sample', required=True, type=str, help='Texture sample.')
    parser.add_argument('--output', required=True, type=str, help='Where to save the generated picture.')
    parser.add_argument('--texture-block-size', required=True, nargs=2, type=int, metavar=('width', 'height'), help='From the texture sample, we will extract blocks of this size.')
    parser.add_argument('--texture-block-count', required=True, type=int, help='How many texture blocks to sample.')
    parser.add_argument('--output-size', nargs=2, required=True, type=int, metavar=('width', 'height'), help='Size of the output picture.')
    parser.add_argument('--overlap', default=1/6, type=float, help='How much neighbouring textures should overlap.')
    args = parser.parse_args()

    (output_width, output_height) = args.output_size
    (block_width, block_height) = args.texture_block_size

    (blocks_per_width, blocks_per_height) = (output_width // block_width, output_height // block_height)

    assert blocks_per_width > 0
    assert blocks_per_height > 0


# Resize the image to fit the blocks completely.
    (output_width, output_height) = (block_width * blocks_per_width, block_height * blocks_per_height)

    sample_img = read_image(args.sample)
    (sample_width, sample_height, nchannels) = sample_img.shape

    params = Parameters(
            output_width=output_width, output_height=output_height,
            block_width=block_width, block_height=block_height,
            blocks_per_width=blocks_per_width, blocks_per_height=blocks_per_height,
            nchannels=nchannels, overlap=args.overlap)

    blocks = []

    for _ in range(args.texture_block_count):
        w = random.randint(0, sample_width - block_width)
        h = random.randint(0, sample_height - block_height)

        block = sample_img[w : w + block_width, h : h + block_height]
        blocks.append(block)

    output = place_overlap_and_edge_cut(params, blocks)
    misc.imsave(args.output, output)


if __name__ == '__main__':
    sys.exit(main())
