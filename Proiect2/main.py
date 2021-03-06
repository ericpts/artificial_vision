#!/usr/bin/python3

import sys
import argparse
import matplotlib.pyplot as plt

import skimage.color
import skimage.filters
from skimage.viewer import ImageViewer
from skimage.viewer.canvastools import RectangleTool

import numpy as np

from scipy import ndarray

from tqdm import tqdm

sys.path.append('../')
from util import *


def get_column_path_fn(strategy):
    if strategy == 'dynamicprogramming':
        return get_column_path_dynamicprogramming
    elif strategy == 'random':
        return get_column_path_random
    elif strategy == 'greedy':
        return get_column_path_greedy
    assert (False)  # Invalid strategy!


get_best_column_path = None


def remove_column_path(img, path):
    (n, m) = img.shape[0:2]
    ret = ndarray((n, m - 1, 3), dtype=img.dtype)
    for i in range(n):
        assert i in path
        j = path[i]
        ret[i] = numpy.concatenate((img[i][0:j], img[i][j + 1:m]), axis=0)
    return ret


def remove_columns(img, cnt):
    print('Removing columns')
    for i in tqdm(range(cnt)):
        energy = image_energy(img)
        path = get_best_column_path(energy)
        img = remove_column_path(img, path)
    return img


def remove_lines(img, cnt):
    print('Removing lines by transposing and removing columns')
    return transpose(remove_columns(transpose(img), cnt))


def debug_path(energy, path):
    (n, m) = energy.shape[0:2]
    dest = ndarray((n, m), dtype=energy.dtype)
    for i in range(n):
        for j in range(m):
            dest[i][j] = energy[i][j]
        j = path[i]
        dest[i][j] = 1
    plt.imshow(dest)
    plt.show()


def add_columns(img, cnt):
    paths = []
    original_img = img
    print('Adding {} columns...'.format(cnt))
    print('Finding {} paths'.format(cnt))
    for i in tqdm(range(cnt)):
        energy = image_energy(img)
        path = get_best_column_path(energy)
        paths.append(path)
        img = remove_column_path(img, path)

    columns_per_row = {}

    for p in paths:
        for (r, c) in p.items():
            if r not in columns_per_row:
                columns_per_row[r] = []
            columns_per_row[r].append(c)

    img = original_img
    (n, m) = img.shape[0:2]

    new_img = ndarray((n, m + cnt, 3), dtype=img.dtype)

    print('Processing rows in order to add columns...')
    for (row, cols) in tqdm(columns_per_row.items()):
        new_row = img[row]

        while len(cols) > 0:
            col = cols[0]
            del cols[0]

            # As the paths that come after this one are affected by the removal of
            # `col`, we have to adjust by an offset.
            for (i, c) in enumerate(cols):
                if c >= col:
                    cols[i] += 2

            neighbours = []

            if col - 1 >= 0:
                neighbours.append(new_row[col - 1])
            else:
                neighbours.append(new_row[col])

            neighbours.append(new_row[col])

            if col + 1 < len(new_row):
                neighbours.append(new_row[col + 1])
            else:
                neighbours.append(new_row[col])

            new_col1 = np.reshape(np.mean(neighbours[0:2], axis=(0,)), (1, 3))
            new_col2 = np.reshape(np.mean(neighbours[1:3], axis=(0,)), (1, 3))

            new_row = np.concatenate((new_row[:col], new_col1, new_col2, new_row[col + 1:]))

        new_img[row] = new_row

    return new_img


def add_lines(img, cnt):
    return transpose(add_columns(transpose(img), cnt))


def remove_object(img, x0, y0, len_x, len_y):
    if len_x < len_y:
        # We have fewer lines than columns.
        return transpose(remove_object(transpose(img), y0, x0, len_y, len_x))

    assert (len_y <= len_x)
    init_len_y = len_y

    print('Removing an object within a bounding box of {}x{}...'.format(len_x, len_y))
    for _ in tqdm(range(init_len_y)):
        energy = image_energy(img)

        # For every line on which the special rectangle appears, mark all columns not within the rectangle with infinite cost.
        # As a result, the path found will surely pass through the rectangle.
        energy[x0:x0 + len_x, :y0].fill(INF)
        energy[x0:x0 + len_x, y0 + len_y:].fill(INF)

        path = get_best_column_path(energy)

        img = remove_column_path(img, path)

        len_y -= 1

    return img


def remove_object_loop(img):
    print("Remove object")

    viewer = ImageViewer(img)

    def on_enter(extens):
        coord = np.int64(extens)

        [x0, y0] = [coord[2], coord[0]]
        [x1, y1] = [coord[3], coord[1]]

        print([x0, y0], [x1, y1])

        (x, y) = (x0, y0)
        (len_x, len_y) = (x1 - x, y1 - y)

        viewer.image = remove_object(viewer.image, x0, y0, len_x, len_y)

    rect_tool = RectangleTool(viewer, on_enter=on_enter)
    viewer.show()
    return viewer.image


def main():
    parser = argparse.ArgumentParser(description='Smartly resize imges.')
    parser.add_argument('target', type=str, help='Target picture to resize.')
    parser.add_argument('output', type=str, help='Where to save the picture.')
    parser.add_argument(
        '--columns',
        type=int,
        help='Desired change in columns. Positive means add, negative means remove.')
    parser.add_argument(
        '--lines',
        type=int,
        help='Desired change in lines. Positive means add, negative means remove.')

    parser.add_argument(
        '--path-algorithm',
        type=str,
        default='dynamicprogramming',
        choices=['random', 'greedy', 'dynamicprogramming'],
        help='Algorithm to use for path selection.')

    parser.add_argument(
        '--remove-object-fixed-coordinates',
        metavar=('x0', 'y0', 'x1', 'y1'),
        nargs=4,
        type=int,
        help='The coordinates of the rectangle to remove.')
    parser.add_argument(
        '--remove-object-mouse',
        action='store_true',
        help='Select the object to be removed by using the mouse.')
    args = parser.parse_args()

    global get_best_column_path
    get_best_column_path = get_column_path_fn(args.path_algorithm)

    img = read_image(args.target)

    fin_img = img

    if args.remove_object_mouse:
        fin_img = remove_object_loop(fin_img)

    if args.remove_object_fixed_coordinates:
        [x0, y0, x1, y1] = args.remove_object_fixed_coordinates
        len_x = x1 - x0
        len_y = y1 - y0
        fin_img = remove_object(fin_img, x0, y0, len_x, len_y)

    if args.columns:
        if args.columns < 0:
            fin_img = remove_columns(fin_img, -args.columns)
        else:
            fin_img = add_columns(fin_img, args.columns)

    if args.lines:
        if args.lines < 0:
            fin_img = remove_lines(fin_img, -args.lines)
        else:
            fin_img = add_lines(fin_img, args.lines)

    misc.imsave(args.output, fin_img)


if __name__ == '__main__':
    sys.exit(main())
