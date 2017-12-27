#!/usr/bin/env python3

import os
import sys

from pathlib import Path
sys.path.append("../")
from util import (read_image, misc, clip_square)
from skimage.transform import resize


def resize_image(img):
    # Resize image to 128x128
    clipped_img = clip_square(img)
    resized_img = resize(clipped_img, (128, 128))
    return resized_img


def generate_training():
    positive = Path('positive_training')
    negative = Path('negative_training')

    os.makedirs(positive, exist_ok=True)
    os.makedirs(negative, exist_ok=True)

    for i in Path('data').iterdir():
        if 'Antrenare' not in str(i):
            continue

        for f in i.iterdir():
            if str(f.name).startswith('pos'):
                dest = positive
            elif str(f.name).startswith('neg'):
                dest = negative
            else:
                sys.exit('Could not determine where to place {}'.format(f))

            misc.imsave(dest / f.name, resize_image(read_image(f)))


def generate_testing():
    positive = Path('positive_testing')
    negative = Path('negative_testing')

    os.makedirs(positive, exist_ok=True)
    os.makedirs(negative, exist_ok=True)

    for i in Path('data').iterdir():
        if 'Testare' not in str(i):
            continue

        if str(i.name).endswith('pozitive'):
            dest = positive
        elif str(i.name).endswith('negative'):
            dest = negative
        else:
            sys.exit('Could not determine where to place {}'.format(f))

        for f in i.iterdir():
            misc.imsave(dest / f.name, resize_image(read_image(f)))


def main():
    generate_training()
    generate_testing()


if __name__ == '__main__':
    main()
