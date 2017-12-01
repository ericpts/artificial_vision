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
GRID_ALLOW_DUPLICATES = False

def split_image(
        target,
        pieces_per_horizontal,
        pieces_per_vertical,
        random_split=False,
        random_coef=5):
    (big_w, big_h) = target.shape[0:2]
    assert big_w % pieces_per_horizontal == 0
    assert big_h % pieces_per_vertical == 0

    small_w = int(big_w / pieces_per_horizontal)
    small_h = int(big_h / pieces_per_vertical)

    def split_into_grid():
        index = 0
        ret = []

        grid = {}
        for i in range(pieces_per_horizontal):
            for j in range(pieces_per_vertical):
                start_w = small_w * i
                start_h = small_h * j

                neighbours = []

                def add_if_there(i, j):
                    if (i, j) in grid:
                        neighbours.append(grid[(i, j)])

                add_if_there(i + 1, j)
                add_if_there(i - 1, j)
                add_if_there(i, j + 1)
                add_if_there(i, j - 1)

                now_target = target[start_w: start_w + small_w, start_h: start_h + small_h]
                now = ImageChunk(index, start_w, start_h, small_w, small_h, now_target, neighbours)
                ret.append(now)

                grid[(i, j)] = now
                index += 1
        return ret

    def split_random():
        print("splitting randomly")
        index = 0
        total_pieces = pieces_per_vertical * pieces_per_horizontal * random_coef
        ret = []
        for _ in range(total_pieces):
            start_w = random.randint(0, big_w - small_w)
            start_h = random.randint(0, big_h - small_h)

            now_target = target[start_w: start_w + small_w, start_h: start_h + small_h]
            ret.append(ImageChunk(index, start_w, start_h, small_w, small_h, now_target))
            index += 1
        return ret

    if random_split:
        return split_random()
    else:
        return split_into_grid()


def get_vertical_piece_count(target, sample, horizontal_pieces_count):
    (big_w, big_h) = target.shape[0:2]
    (small_w, small_h) = sample.shape[0:2]
    new_big_w = small_w * horizontal_pieces_count
    new_big_h = (big_h / big_w) * new_big_w
    vertical_pieces_count = int(new_big_h / small_h)
    return vertical_pieces_count


def resize_target(target, sample, pieces_per_horizontal, pieces_per_vertical):
    (small_w, small_h) = sample.shape[0:2]
    (big_w, big_h) = (pieces_per_horizontal * small_w, pieces_per_vertical * small_h)
    return misc.imresize(target, (big_w, big_h))


def solve_for_chunks(chunks, kd_tree, allow_duplicates):
    def choose_best_unique(chunk):
        candidates = find_in_kd_tree(chunk.data, kd_tree, neighbours_count=4)
        for cand in candidates:
            ok = True
            for nbr in c.neighbours:
                if nbr.chosen == cand:
                    ok = False
            if ok:
                return cand
        assert False

    for c in chunks:
        color = average_color(c.data)
        chosen = None

        if allow_duplicates:
            chosen = find_in_kd_tree(c.data, kd_tree, neighbours_count=1)
        else:
            chosen = choose_best_unique(c)

        c.chosen = chosen


def render_chunks(chunks, collection):
    width = max([c.end_w for c in chunks])
    height = max([c.end_h for c in chunks])

    output = ndarray(shape=(width, height, 3), dtype=int)
    for c in chunks:
        output[c.start_w: c.end_w, c.start_h: c.end_h] = collection[c.chosen]
    return output


def read_collection(collection_dir):
    images = []
    for x in os.listdir(collection_dir):
        images.append(read_image("{}{}".format(collection_dir, x)))
    return images


def average_color(img):
    ret = numpy.mean(img, axis=(0, 1))
    if isinstance(ret, numpy.float64):
        # Convert from black and white to RGB.
        return [ret, ret, ret]
    return ret


def make_kd_tree(collection):
    collection_colors = [average_color(x) for x in collection]
    return KDTree(collection_colors)


def find_in_kd_tree(target, kd_tree, neighbours_count=1):
    return kd_tree.query(average_color(target), k=neighbours_count)[1]


class ImageChunk(object):
    def __init__(self, index, start_w, start_h, len_w, len_h, data, neighbours=[]):
        self.index = index
        self.start_w = start_w
        self.start_h = start_h
        self.len_w = len_w
        self.len_h = len_h
        self.data = data
        self.neighbours = neighbours

        self.chosen = None

        self.end_w = self.start_w + self.len_w
        self.end_h = self.start_h + self.len_h


def unpickle(file):
    import pickle
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
    return dict


def read_cifar(cifar_dir):
    label_names = unpickle(str(Path(cifar_dir) / 'batches.meta'))

    def read_batch(n):
        f = str(Path(cifar_dir) / 'data_batch_{}'.format(n))
        raw = unpickle(f)
        return (raw[b'labels'], raw[b'data'])

    labels = []
    data = ndarray((0, 3072), dtype=int)
    for (cur_label, cur_data) in map(read_batch, range(1, 6)):
        labels += cur_label
        data = numpy.concatenate((data, cur_data), axis=0)

    def cifar_data_row_to_image(row):
        w = int(math.sqrt(len(row) / 3))
        h = w
        row = numpy.reshape(row, (3, w, h))
        row = numpy.rollaxis(row, 0, 3)
        return row

    d = {
        'labels': labels,
        'images': [cifar_data_row_to_image(r) for r in data],
    }

    return ([l.decode('utf-8') for l in label_names[b'label_names']], d)


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
        chunks = split_image(
            target,
            horizontal_pieces_count,
            vertical_pieces_count,
            random_split=True)
    else:
        chunks = split_image(target, horizontal_pieces_count, vertical_pieces_count)

    solve_for_chunks(chunks, kd_tree, args.allow_duplicates)

    return render_chunks(chunks, collection)


def main():
    parser = argparse.ArgumentParser(
        description='Mimic a picture by creating a grid made out of sample pictures.')
    parser.add_argument(
        '--collection_dir',
        dest='collection_dir',
        default=COLLECTION_DIR,
        type=str,
        help='Where to find the project base pictures')
    parser.add_argument('--cifar_dir', dest='cifar_dir', default=CIFAR_DIR,
                        type=str, help='Where to find the project CIFAR pictures')
    parser.add_argument(
        '--random',
        dest='random',
        action='store_true',
        default=PLACE_RANDOM,
        help='Whether to place pictures randomly')
    parser.add_argument(
        '--allow_duplicates',
        action='store_true',
        dest='allow_duplicates',
        default=GRID_ALLOW_DUPLICATES,
        help='Whether to allow adjacent duplicates')
    parser.add_argument(
        '--horizontal_pieces',
        dest='horizontal_pieces',
        type=int,
        default=HORIZONTAL_PIECES,
        help='How many pieces to plaze horizontally')
    parser.add_argument(
        '--use_cifar',
        dest='use_cifar',
        action='store_true',
        default=False,
        help='Whether to use the cifar pictures for construction')
    parser.add_argument('--cifar_label', dest='cifar_label', type=str,
                        help='What type of cifar pictures to use')
    parser.add_argument('target', type=str, help='Target picture to recreate')
    parser.add_argument('output', type=str, help='Where to store the generated picture')

    args = parser.parse_args()
    misc.imsave(args.output, generate_image_in_memory(args))


if __name__ == "__main__":
    sys.exit(main())
