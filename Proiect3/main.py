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
from parameters import Parameters

from place_overlap_and_edge_cut import place_overlap_and_edge_cut
from place_overlap import place_overlap
from place_random import place_random

def main():

    parser = argparse.ArgumentParser(description = 'Texture generator.')
    parser.add_argument('--sample', required=True, type=str, help='Texture sample.')
    parser.add_argument('--output', required=True, type=str, help='Where to save the generated picture.')
    parser.add_argument('--texture-block-size', required=True, nargs=2, type=int, metavar=('width', 'height'), help='From the texture sample, we will extract blocks of this size.')
    parser.add_argument('--texture-block-count', required=True, type=int, help='How many texture blocks to sample.')
    parser.add_argument('--output-size', nargs=2, required=True, type=int, metavar=('width', 'height'), help='Size of the output picture.')
    parser.add_argument('--overlap', default=1/6, type=float, help='How much neighbouring textures should overlap.')
    parser.add_argument('--algorithm', required=True, type=str, choices=['random' , 'overlap', 'overlap-and-cut'], help='Which algorithm to use.')
    args = parser.parse_args()

    (output_width, output_height) = args.output_size
    (block_width, block_height) = args.texture_block_size

    (blocks_per_width, blocks_per_height) = (
            int(1 + (output_width - block_width) // (block_width - args.overlap * block_width)),
            int(1 + (output_height - block_height) // (block_height - args.overlap * block_height)))

    assert blocks_per_width > 0
    assert blocks_per_height > 0

# Resize the image to fit the blocks completely.
    (output_width, output_height) = (block_width * blocks_per_width, block_height * blocks_per_height)

    sample_img = read_image(args.sample)
    (sample_height, sample_width, nchannels) = sample_img.shape

    params = Parameters(
            output_height=output_height,
            output_width=output_width,
            block_height=block_height,
            block_width=block_width,
            blocks_per_height=blocks_per_height,
            blocks_per_width=blocks_per_width,
            nchannels=nchannels, overlap=args.overlap)

    blocks = []

    for _ in range(args.texture_block_count):
        h = random.randint(0, sample_height - block_height)
        w = random.randint(0, sample_width - block_width)

        block = sample_img[h : h + block_height, w : w + block_width]
        blocks.append(block)

    if args.algorithm == 'random':
        output = place_random(params, blocks)
    elif args.algorithm == 'overlap':
        output = place_overlap(params, blocks)
    elif args.algorithm == 'overlap-and-cut':
        output = place_overlap_and_edge_cut(params, blocks)
    else:
        print('Invalid algorithm: {}'.format(algorithm))
        sys.exit(-1)
    misc.imsave(args.output, output)

if __name__ == '__main__':
    main()
