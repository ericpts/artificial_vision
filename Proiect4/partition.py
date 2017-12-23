#!/usr/bin/python3

import os
import sys
import shutil

from pathlib import Path


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

      shutil.copy(f, dest)


if __name__ == '__main__':
  main()
