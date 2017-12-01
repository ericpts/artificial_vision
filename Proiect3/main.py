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
from transfer_texture import transfer_texture

def check_args(args):
    if args.algorithm == 'texture-transfer':
        if not args.transfer_image:
            print('algorithm texture-transfer requires --transfer-image')
            sys.exit(-1)

        if args.output_size:
            print('algorithm texture-transfer cannot be given a fixed output size')
            sys.exit(-2)

        args.output_size = read_image(args.transfer_image).shape[0:2]

    if args.algorithm != 'texture-transfer':
        if not args.output_size:
            print('algorithm {} requires --output-size'.format(args.algorithm))
            sys.exit(-3)

def main():

    parser = argparse.ArgumentParser(description='Texture generator.')
    parser.add_argument('--sample', required=True, type=str, help='Texture sample.')
    parser.add_argument('--output', required=True, type=str,
                        help='Where to save the generated picture.')
    parser.add_argument(
        '--texture-block-size',
        required=True,
        nargs=2,
        type=int,
        metavar=(
            'width',
            'height'),
        help='From the texture sample, we will extract blocks of this size.')
    parser.add_argument(
        '--texture-block-count',
        required=True,
        type=int,
        help='How many texture blocks to sample.')
    parser.add_argument(
        '--output-size',
        nargs=2,
        type=int,
        default=None,
        metavar=(
            'height',
            'width'),
        help='Size of the output picture.')
    parser.add_argument(
        '--overlap', default=1 / 6, type=float,
        help='How much neighbouring textures should overlap.')
    parser.add_argument(
        '--algorithm',
        required=True,
        type=str,
        choices=[
            'random',
            'overlap',
            'overlap-and-cut',
            'texture-transfer'],
        help='Which algorithm to use.')
    parser.add_argument(
        '--transfer-image',
        type=str,
        help='Which picture to transfer texture onto.')
    parser.add_argument(
        '--transfer-coefficient',
        type=float,
        metavar='alpha',
        default=0.5,
        help='Alpha weight will be given to texture fitness and (1 - alpha) to correspondence fitness.')
    args = parser.parse_args()

    check_args(args)

    (output_height, output_width) = args.output_size
    (block_height, block_width) = args.texture_block_size

    height_overlap = int(args.overlap * block_height)
    width_overlap = int(args.overlap * block_width)

    (blocks_per_height, blocks_per_width) = (
        int(1 + (output_height - block_height) // (block_height - height_overlap)),
        int(1 + (output_width - block_width) // (block_width - width_overlap))
        )

# Sanity checks.
    assert blocks_per_height > 0
    assert blocks_per_width > 0

# The last blocks should fit completely within the image.
    assert (blocks_per_height - 1) * (block_height - height_overlap) + block_height < output_height
    assert (blocks_per_width - 1) * (block_width - width_overlap) + block_width < output_width

    # pdb.set_trace()

# We should not be able to add any more blocks.
    assert (blocks_per_height - 0) * (block_height - height_overlap) + block_height >= output_height
    assert (blocks_per_width - 0) * (block_width - width_overlap) + block_width >= output_width

# Resize the image to fit the blocks completely.
    # (output_width, output_height) = (block_width * blocks_per_width, block_height * blocks_per_height)

    sample_img = read_image(args.sample)
    (sample_height, sample_width, nchannels) = sample_img.shape

    params = Parameters(
        output_height=output_height,
        output_width=output_width,
        block_height=block_height,
        block_width=block_width,
        blocks_per_height=blocks_per_height,
        blocks_per_width=blocks_per_width,
        nchannels=nchannels,
        overlap=args.overlap,
        transfer_coefficient=args.transfer_coefficient)

    blocks = []

    for _ in range(args.texture_block_count):
        h = random.randint(0, sample_height - block_height)
        w = random.randint(0, sample_width - block_width)

        block = sample_img[h: h + block_height, w: w + block_width]
        blocks.append(block)

    if args.algorithm == 'random':
        output = place_random(params, blocks)
    elif args.algorithm == 'overlap':
        output = place_overlap(params, blocks)
    elif args.algorithm == 'overlap-and-cut':
        output = place_overlap_and_edge_cut(params, blocks)
    elif args.algorithm == 'texture-transfer':
        output = transfer_texture(params, blocks, read_image(args.transfer_image))
    else:
        print('Invalid algorithm: {}'.format(args.algorithm))
        sys.exit(-1)
    misc.imsave(args.output, output)


if __name__ == '__main__':
    main()
