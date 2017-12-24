#!/usr/bin/env python3

import pdb
import sys
import time
import argparse
import random
import itertools
import functools
import matplotlib.pyplot as plt
from itertools import product
from math import pi
import json
import yaml
from pathlib import Path

from typing import List, Tuple

import skimage.color
import skimage.filters
from skimage.feature import hog
from skimage.viewer import ImageViewer
from skimage.viewer.canvastools import RectangleTool

import numpy as np

import scipy.spatial.distance
from scipy import ndarray
from scipy.ndimage import convolve

from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans

from tqdm import tqdm

sys.path.append('../')
from util import *

from params import Parameters


def get_classifier(params: Parameters):
    if params.classifier == 'SVM':
        return svm.SVC()
    elif params.classifier == 'NN':
        return KNeighborsClassifier(n_neighbors=1)

def image_points(image: ndarray, params: Parameters) -> List:
    (h, w) = image.shape[0:2]

    ys = np.linspace(params.border, h - params.border, params.points_on_height, dtype=int)
    xs = np.linspace(params.border, w - params.border, params.points_on_width, dtype=int)

    return list(product(xs, ys))


def patch_around_point(image: ndarray, params: Parameters, point: Tuple) -> ndarray:
    h = (params.cells_per_patch * params.cell_size)
    w = h # It's always a square.

    up = h // 2
    down = h - up

    left = w // 2
    right = w - left

    (x, y) = point

    return image[x - up: x + down, y - left: y + right]

def show(i):
    plt.imshow(i)
    plt.show()

def gradient_for_patch(patch: ndarray) -> ndarray:
    horiz_kernel = np.reshape(np.asarray([1, 0, -1]), (1, 3))
    vert_kernel = np.transpose(horiz_kernel)

    gray = to_grayscale(patch)

    h = np.ravel(convolve(gray, horiz_kernel))
    v = np.ravel(convolve(gray, vert_kernel))

    d2 = np.stack((h, v), axis=1)
    return np.ndarray(shape=(d2.shape[0]), buffer=d2, dtype=complex)


def descriptor_for_patch(patch: ndarray, params: Parameters) -> ndarray:
    cells = [x for s in np.split(patch, params.cells_per_patch, axis=0) for x in np.split(s, params.cells_per_patch, axis=1)]

    gradients = [gradient_for_patch(c) for c in cells]

    angles = np.angle(gradients)
    buckets = np.floor(((angles + pi) / (2 * pi) - sys.float_info.epsilon) * params.bin_size).astype(int)
    lengths = np.absolute(gradients)

    histogram = np.zeros(shape=(len(gradients), params.bin_size), dtype=float)

    histogram[:, buckets] += lengths

    norms = normalize(histogram, norm='l1')

    return np.ravel(norms) # Return a single vector of features.


def hogs_for_image(image: ndarray, params: Parameters) -> ndarray:
    points = image_points(image, params)
    patches = [patch_around_point(image, params, point) for point in points]

    descriptors = [descriptor_for_patch(p, params) for p in patches]

    return descriptors

def generate_vocabulary(hogs: List[ndarray], params: Parameters) -> KMeans:
    kmeans = KMeans(n_clusters=params.clusters).fit(hogs)
    return kmeans

def features_for_hogs(hogs: List[ndarray], vocabulary: KMeans) -> ndarray:
    pred = vocabulary.predict(hogs)
    (indexes, features) = np.unique(pred, return_counts=True)

    ret = np.zeros(shape=(vocabulary.n_clusters), dtype=int)
    ret[indexes] += features
    return ret


def generate_data(vocabulary: KMeans, params: Parameters, kind: str) -> Tuple[ndarray, List]:
    if kind == 'testing':
        positive_dir = params.positive_testing_dir
        negative_dir = params.negative_testing_dir
    elif kind == 'training':
        positive_dir = params.positive_training_dir
        negative_dir = params.negative_training_dir
    else:
        raise ValueError('Unknown data kind: {}. It should be one of testing or training.'.format(kind))

    def features_for_dir(path: str):
        return [features_for_hogs(hogs_for_image(read_image(f), params), vocabulary) for f in Path(path).iterdir()]

    positive_features = features_for_dir(positive_dir)
    negative_features = features_for_dir(negative_dir)

    features = positive_features + negative_features
    labels = [1] * len(positive_features) + [-1] * len(negative_features)

    return (features, labels)

def main():
    parser = argparse.ArgumentParser(description='Car recogniser.')

    parser.add_argument(
        '-c', '--configuration', default='conf.yml', type=str, help='Configuration yaml file.')

    args = parser.parse_args()

    if args.configuration:
        p = Path(args.configuration)
        if not p.exists():
            raise OSError('Configuration file {} does not exist'.format(args.configuration))

        with p.open() as f:
            conf = yaml.load(f)

    params = Parameters(**conf)
    hogs = [hog for f in itertools.chain(Path(params.positive_training_dir).iterdir(), Path(params.negative_training_dir).iterdir()) for hog in hogs_for_image(read_image(f), params)]
    vocabulary = generate_vocabulary(hogs, params)

    def cross_validate(samples, labels):
        X_train, X_test, y_train, y_test = train_test_split(
            samples, labels, test_size=0.4, random_state=0)

        print(len(X_train), len(X_test))
        print(len(y_train), len(y_test))

        clf = get_classifier(params).fit(X_train, y_train)
        return clf.score(X_test, y_test)

    samples, labels = generate_data(vocabulary, params, kind='training')
    clf = get_classifier(params).fit(samples, labels)

    testing_samples, testing_labels = generate_data(vocabulary, params, kind='testing')
    print(clf.score(testing_samples, testing_labels))


if __name__ == '__main__':
    main()
