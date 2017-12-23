#!/usr/bin/env python3

import pdb
import sys
import time
import argparse
import random
import itertools
import functools
import matplotlib.pyplot as plt
import json
import yaml
from pathlib import Path

from typing import List

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

from tqdm import tqdm

sys.path.append('../')
from util import *

from params import Parameters


def get_classifier(params: Parameters):
    if params.classifier == 'SVM':
        return svm.SVC()
    elif params.classifier == 'NN':
        return KNeighborsClassifier(n_neighbors=1)


def image_descriptor(image: ndarray, params: Parameters) -> ndarray:
    """ Given a image, generate its' feature descriptor. """

    (height, width) = image.shape[0:2]
    (image_height, image_width) = (height // params.points_on_height,
                                   width // params.points_on_width)

    return hog(
        to_grayscale(image),
        pixels_per_cell=(image_height, image_width),
        block_norm='L2-Hys',
        cells_per_block=(params.cells_per_block, params.cells_per_block),
        visualise=False)


def feature_vector_for_image(image: ndarray, params: Parameters) -> ndarray:
    ret = image_descriptor(image, params)
    # plt.imshow(img)
    # plt.show()
    # print(ret.shape)
    return ret


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

    positive_samples = [
        feature_vector_for_image(read_image(f), params)
        for f in Path(params.positive_training_dir).iterdir()
    ]
    negative_samples = [
        feature_vector_for_image(read_image(f), params)
        for f in Path(params.negative_training_dir).iterdir()
    ]

    samples = positive_samples + negative_samples
    labels = [1] * len(positive_samples) + [-1] * len(negative_samples)

    def cross_validate(samples, labels):
        X_train, X_test, y_train, y_test = train_test_split(
            samples, labels, test_size=0.4, random_state=0)

        print(len(X_train), len(X_test))
        print(len(y_train), len(y_test))

        clf = get_classifier(params).fit(X_train, y_train)
        return clf.score(X_test, y_test)

    clf = get_classifier(params).fit(samples, labels)

    positive_testing = [
        feature_vector_for_image(read_image(f), params)
        for f in Path(params.positive_testing_dir).iterdir()
    ]
    negative_testing = [
        feature_vector_for_image(read_image(f), params)
        for f in Path(params.negative_testing_dir).iterdir()
    ]

    testing_samples = positive_testing + negative_testing
    testing_labels = [1] * len(positive_testing) + [-1] * len(negative_testing)

    print(clf.score(testing_samples, testing_labels))


if __name__ == '__main__':
    main()
