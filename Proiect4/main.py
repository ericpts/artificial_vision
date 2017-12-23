#!/usr/bin/python3

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
from params import Parameters

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


def partition_image(image: ndarray, params: Parameters) -> List[ndarray]:
  """ Split the given image into patches - "words", which will be processed by a descriptor to produce features for training. """
  (height, width) = image.shape[0:2]

  (patch_height, patch_width) = (height // params.points_on_height, width // params.points_on_width)

  ret = []

  h = 0
  for i in range(params.points_on_height):
    next_h = min(h + patch_height, height)

    w = 0
    for j in range(params.points_on_width):
      next_w = min(w + patch_width, width)
      ret.append(image[h:next_h, w:next_w])

      w = next_w

    h = next_h
  return ret


def patch_descriptor(patch: ndarray) -> ndarray:
  """ Given a patch, generate its' feature descriptor. """
  pass


def main():
  parser = argparse.ArgumentParser(description='Car recogniser.')

  parser.add_argument(
      '-c', '--configuration', required=True, type=str, help='Configuration yaml file.')

  args = parser.parse_args()

  if args.configuration:
    p = Path(args.configuration)
    if not p.exists():
      raise OSError('Configuration file {} does not exist'.format(args.configuration))

    with p.open() as f:
      conf = yaml.load(f)

  params = Parameters(**conf)

  positive_samples = [[patch_descriptor(patch)
                       for patch in partition_image(read_image(f), params)]
                      for f in Path(params.positive_training_dir).iterdir()]
  negative_samples = [[patch_descriptor(patch)
                       for patch in partition_image(read_image(f), params)]
                      for f in Path(params.negative_training_dir).iterdir()]


if __name__ == '__main__':
  main()
