#!/usr/bin/python3

import os
import sys
import shutil

from pathlib import Path
sys.path.append("../")
from util import (read_image, misc, clip_square)
from skimage.transform import resize

def main():

  positive = Path('positive')
  negative = Path('negative')

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
      
      img = read_image(f)
      # Resize image to 128x128
      clipped_img = clip_square(img)
      resized_img = resize(clipped_img, (128, 128))
      
      misc.imsave(dest / f.name, resized_img)


if __name__ == '__main__':
  main()
