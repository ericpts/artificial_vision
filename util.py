import numpy
import time
import skimage.io
from scipy.spatial.distance import euclidean
from scipy.spatial import KDTree
from scipy import ndarray
from scipy import misc
from pathlib import Path
import math
import sys
import matplotlib.pyplot as plt
import os
import pdb
import random
import numpy as np

INF = 2**60


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


def read_image(file_name):
    return normalize_image(skimage.io.imread(file_name))
    # return normalize_image(plt.imread(file_name))


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


def normalize_image(img):
    ret = misc.imresize(img, 100)
    if len(ret.shape) == 3 and ret.shape[2] == 4:
        # Transform from RGBA to RGB.
        ret = np.delete(ret, axis=2, obj=3)
    return ret


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


def neighbours_with_limits(n, m, i, j):
    assert i > 0

    if j == 0:
        return [(i - 1, j), (i - 1, j + 1)]

    if j == m - 1:
        return [(i - 1, j - 1), (i - 1, j)]

    return [
        (i - 1, j - 1),
        (i - 1, j),
        (i - 1, j + 1)
    ]


def get_column_path_dynamicprogramming(energy):
    """ Returns the path, a dict from line to column. """
    (n, m) = energy.shape[0:2]

    def neighbours(i, j):
        return neighbours_with_limits(n, m, i, j)

    best = ndarray(energy.shape, dtype=energy.dtype)

    for j in range(m):
        best[0][j] = energy[0][j]

    for i in range(1, n):
        shift_left = np.append(best[i - 1][1:], INF)
        shift_right = np.append([INF], best[i - 1][:-1])

        best[i] = energy[i] + np.minimum(
            best[i - 1],
            np.minimum(shift_left,
                       shift_right)
        )

    starting = 0
    for j in range(m):
        if (best[n - 1][j] < best[n - 1][starting]):
            starting = j

    def get_prev(i, j):
        for nbr in neighbours(i, j):
            if best[nbr] + energy[i][j] == best[i][j]:
                return nbr

    path = {}

    def reconstruct_path(i, j):
        path[i] = j
        while i > 0:
            (i, j) = get_prev(i, j)
            path[i] = j

    reconstruct_path(n - 1, starting)

    return path


def get_column_path_random(energy):
    (n, m) = energy.shape[0:2]

    def neighbours(i, j):
        return neighbours_with_limits(n, m, i, j)

    path = {}
    i = n - 1
    j = random.randint(0, m - 1)

    while i > 0:
        path[i] = j
        (i, j) = random.choice(neighbours(i, j))
    path[i] = j
    return path


def get_column_path_greedy(energy):
    (n, m) = energy.shape[0:2]

    def neighbours(i, j):
        return neighbours_with_limits(n, m, i, j)

    path = {}
    i = n - 1
    j = np.argmin(energy[i])

    while i > 0:
        path[i] = j
        nbrs = neighbours(i, j)

        best_nbr = None
        for n in nbrs:
            if best_nbr is None or energy[n] < energy[best_nbr]:
                best_nbr = n

        (i, j) = best_nbr

    path[i] = j
    return path


def transpose(img):
    if len(img.shape) == 3:
        return np.transpose(img, axes=(1, 0, 2))
    elif len(img.shape) == 2:
        return np.transpose(img, axes=(1, 0))

    assert False
