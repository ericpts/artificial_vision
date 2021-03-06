#!/usr/bin/env python3

import sys
import pdb
import argparse
import itertools
import random
import matplotlib.pyplot as plt
from itertools import product
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

from sklearn.manifold import TSNE
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.feature_extraction.image import extract_patches_2d

sys.path.append('../')
from util import *

from params import Parameters

def show(image: ndarray):
    plt.imshow(image)
    plt.show()

def features_for_image(image: ndarray, params: Parameters, feature_vector: bool = True) -> ndarray:
    fd = hog(to_grayscale(image), block_norm='L1', orientations=params.bin_size, pixels_per_cell=(params.cell_size, params.cell_size), cells_per_block=(params.cells_per_block, params.cells_per_block), feature_vector=feature_vector)
# Hack to undo the normalization.
    return fd

def partition_negative_image(image: ndarray, params: Parameters) -> List[ndarray]:
    k = params.window_size
    (n, m) = image.shape[0:2]

    npatches = min(params.samples_per_negative_image, (n - k) * (m - k))
    patches = extract_patches_2d(image, (k, k), max_patches=npatches)

    return patches

def detections_for_image(image: ndarray, params: Parameters, classifier) -> List:

    image = to_grayscale(image)

    k = params.window_size // params.cell_size - params.cells_per_block + 1
    for scale in map(float, params.test_image_scales):
        scaled_img = misc.imresize(image, scale)
        (h, w) = scaled_img.shape[0:2]

        features = features_for_image(scaled_img, params, feature_vector=False)

        (n, m) = features.shape[0:2]

        for i in range(n - k):
            for j in range(m - k):
                window = np.ravel(features[i : i + k, j : j + k])

                r = classifier.decision_function([window])
                face = classifier.predict([window])

                x = int(i / n * h)
                y = int(j / m * w)

                if face[0] == 1 and r[0] >= 1.0:
                    print(r)
                    show(scaled_img[x : x + params.window_size, y : y + params.window_size])


                # print(classifier.predict([window]))
                # print(r)


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

    positive_features = np.asarray([ features_for_image(read_image(f), params) for f in Path(params.positive_training_dir).iterdir() ])
    negative_features = np.asarray([ features_for_image(img, params) for f in Path(params.negative_training_dir).iterdir() for img in partition_negative_image(read_image(f), params) ])

    features = np.concatenate((positive_features, negative_features))
    labels = np.asarray([1] * len(positive_features) + [-1] * len(negative_features))

    print('Training the classifier with {} positives samples and {} negative samples'.format(len(positive_features), len(negative_features)))
    csf = svm.LinearSVC(verbose=True).fit(features, labels)

    def train_for_hard_negatives():
        def false_positives() -> ndarray:
            pred = csf.predict(negative_features)
            fp = [i for i in range(len(negative_features)) if pred[i] != -1]
            return np.asarray(fp)

        fp = false_positives()

        print('Running hard negative samples', end='')
        while len(fp) > 0:

            features = np.concatenate((features, negative_features[fp]))
            labels = np.concatenate((labels, [-1] * len(fp)))

            csf.fit(features, labels)
            fp = false_positives()

            print('.', end='', flush=True)

    def cross_validate(samples, labels):
        X_train, X_test, y_train, y_test = train_test_split(
            samples, labels, test_size=0.4, random_state=0)

        clf = svm.LinearSVC().fit(X_train, y_train)
        return clf.score(X_test, y_test)


    print('cross validation performance: ', (cross_validate(features, labels)))

    for f in Path(params.test_dir).iterdir():
        detections_for_image(read_image(f), params, csf)



if __name__ == '__main__':
    main()
