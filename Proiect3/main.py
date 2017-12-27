#!/usr/bin/python3

import sys
import argparse
import itertools
import matplotlib.pyplot as plt

import skimage.color
import skimage.filters
from skimage.viewer import ImageViewer
from skimage.viewer.canvastools import RectangleTool


import numpy as np

import scipy.spatial.distance
from scipy import ndarray
from scipy.ndimage import convolve

from tqdm import tqdm

sys.path.append('../')
from util import *
from parameters import Parameters

from place_utils import generate_blocks

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

        if args.transfer_niterations == 1 and args.transfer_coefficient is None:
            print('--transfer-niterations 1 requires --transfer-coefficient')
            sys.exit(-4)

        if args.transfer_niterations > 1 and args.transfer_coefficient is not None:
            print(
                '--transfer-niterations {} adjusts its\' own coefficient, so --transfer-coefficient must not be present'.
                format(args.transfer_niterations))
            sys.exit(-5)

        args.output_size = read_image(args.transfer_image).shape[0:2]

    if args.algorithm != 'texture-transfer':
        if not args.output_size:
            print('algorithm {} requires --output-size'.format(args.algorithm))
            sys.exit(-3)


def main():

    parser = argparse.ArgumentParser(description='Texture generator.')
    parser.add_argument('--sample', required=True, type=str, help='Texture sample.')
    parser.add_argument(
        '--output', required=True, type=str, help='Where to save the generated picture.')
    parser.add_argument(
        '--texture-block-size',
        required=True,
        nargs=2,
        type=int,
        metavar=('width', 'height'),
        help='From the texture sample, we will extract blocks of this size.')
    parser.add_argument(
        '--texture-block-count', required=True, type=int, help='How many texture blocks to sample.')
    parser.add_argument(
        '--output-size',
        nargs=2,
        type=int,
        default=None,
        metavar=('height', 'width'),
        help='Size of the output picture.')
    parser.add_argument(
        '--overlap',
        default=1 / 6,
        type=float,
        help='How much neighbouring textures should overlap.')
    parser.add_argument(
        '--algorithm',
        required=True,
        type=str,
        choices=['random', 'overlap', 'overlap-and-cut', 'texture-transfer'],
        help='Which algorithm to use.')
    parser.add_argument(
        '--transfer-image', type=str, help='Which picture to transfer texture onto.')
    parser.add_argument(
        '--transfer-coefficient',
        type=float,
        metavar='alpha',
        default=None,
        help=
        'Alpha weight will be given to texture fitness and (1 - alpha) to correspondence fitness.')
    parser.add_argument(
        '--transfer-niterations',
        type=int,
        metavar='niters',
        default=1,
        help='How many iterations to perform when transferring texture.')
    args = parser.parse_args()

    check_args(args)

    (init_output_height, init_output_width) = args.output_size
    (block_height, block_width) = args.texture_block_size

    sample_img = read_image(args.sample)
    nchannels = sample_img.shape[2]

    params = Parameters(
        texture_block_count=args.texture_block_count,
        init_output_height=init_output_height,
        init_output_width=init_output_width,
        block_height=block_height,
        block_width=block_width,
        nchannels=nchannels,
        overlap=args.overlap,
        transfer_coefficient=args.transfer_coefficient,
        transfer_niterations=args.transfer_niterations)

    if args.algorithm == 'random':
        output = place_random(params, sample_img)
    elif args.algorithm == 'overlap':
        output = place_overlap(params, sample_img)
    elif args.algorithm == 'overlap-and-cut':
        output = place_overlap_and_edge_cut(params, sample_img)
    elif args.algorithm == 'texture-transfer':
        output = transfer_texture(params, sample_img, read_image(args.transfer_image))
    else:
        print('Invalid algorithm: {}'.format(args.algorithm))
        sys.exit(-1)
    misc.imsave(args.output, output)


if __name__ == '__main__':
    main()
