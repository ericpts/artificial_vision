#!/usr/bin/python3

import argparse
import numpy
import time
from scipy.spatial.distance import euclidean
from scipy import ndarray
from scipy import misc
from pathlib import Path
import math
import sys
import matplotlib.pyplot as plt
import os
import pdb
import random

sys.path.append('../')
from util import *

COLLECTION_DIR = "./data/colectie/"
CIFAR_DIR = "./cifar-10-batches-py/"
HORIZONTAL_PIECES = 100
PLACE_RANDOM = False
GRID_ALLOW_DUPLICATES=False

def generate_image_in_memory(args):
    if args.use_cifar:
        print('Using cifar')
        (label_names, dict) = read_cifar(args.cifar_dir)
        labels = dict['labels']
        images = dict['images']

        if args.cifar_label not in label_names:
            print("Cifar_label must be one of {}".format(label_names))
            sys.exit(-1)

        collection = []
        for (label_index, image) in zip(labels, images):
            label = label_names[label_index]
            if label == args.cifar_label:
                collection.append(image)
    else:
        collection = read_collection(args.collection_dir)

    target = read_image(args.target)
    sample = collection[0]
    kd_tree = make_kd_tree(collection)
    output_file = args.output

    horizontal_pieces_count = args.horizontal_pieces
    vertical_pieces_count = get_vertical_piece_count(target, sample, horizontal_pieces_count)
    target = resize_target(target, sample, horizontal_pieces_count, vertical_pieces_count)

    print("Target.shape = {}".format(target.shape))
    print("Collection.shape = {}".format(collection[0].shape))

    if args.random:
        chunks = split_image(target, horizontal_pieces_count, vertical_pieces_count, random_split=True)
    else:
        chunks = split_image(target, horizontal_pieces_count, vertical_pieces_count)

    solve_for_chunks(chunks, kd_tree, args.allow_duplicates)

    return render_chunks(chunks, collection)

def main():
    parser = argparse.ArgumentParser(description = 'Mimic a picture by creating a grid made out of sample pictures.')
    parser.add_argument('--collection_dir', dest='collection_dir', default=COLLECTION_DIR, type=str, help='Where to find the project base pictures')
    parser.add_argument('--cifar_dir', dest='cifar_dir', default=CIFAR_DIR, type=str, help='Where to find the project CIFAR pictures')
    parser.add_argument('--random', dest='random', action='store_true', default=PLACE_RANDOM, help='Whether to place pictures randomly')
    parser.add_argument('--allow_duplicates', action='store_true', dest='allow_duplicates', default=GRID_ALLOW_DUPLICATES, help='Whether to allow adjacent duplicates')
    parser.add_argument('--horizontal_pieces', dest='horizontal_pieces', type=int, default=HORIZONTAL_PIECES, help='How many pieces to plaze horizontally')
    parser.add_argument('--use_cifar', dest='use_cifar', action='store_true', default=False, help='Whether to use the cifar pictures for construction')
    parser.add_argument('--cifar_label', dest='cifar_label', type=str, help='What type of cifar pictures to use')
    parser.add_argument('target', type=str, help='Target picture to recreate')
    parser.add_argument('output', type=str, help='Where to store the generated picture')

    args = parser.parse_args()
    misc.imsave(args.output, generate_image_in_memory(args))

if __name__ == "__main__":
    sys.exit(main())
