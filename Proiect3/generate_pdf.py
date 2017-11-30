#!/usr/bin/python3

from os.path import basename, exists
import sys
import glob
import shutil
import argparse
import subprocess
import tempfile
import random
import urllib
import multiprocessing as mp
from pathlib import Path
from reportlab.pdfgen import canvas

sys.path.append('../')
from util import read_image

from scipy import misc
from lib_google_img.google_images_download import google_images_download

def add_picture_to_canvas(c, pic, desc):
    (w, h) = read_image(pic).shape[0:2]
    c.setPageSize((max(h, 500), w + 100))
    c.drawInlineImage(str(pic), 0, 0)
    c.drawCentredString(h / 2, w + 50, desc)
    c.showPage()


def add_task_1_1(c, tmp_dir_path: Path, data_dir: Path):
    for f in glob.glob(str(data_dir / 'img*.png')):
        fpath = Path(f)
        out = tmp_dir_path / '{}_out.jpg'.format(fpath.stem)
        subprocess.check_call(
                ['python3', 'main.py',
                    '--sample', f,
                    '--output', str(out),
                    '--overlap', str(0.16),
                    '--texture-block-size', *['50', '50'],
                    '--output-size', *['500', '500'],
                    '--texture-block-count', str(1000)])
        add_picture_to_canvas(c, out, fpath.stem)

def main():
    parser = argparse.ArgumentParser(description = 'Generate project pdf.')
    parser.add_argument('output', type=str, help='Where to store the generated pdf.')
    parser.add_argument('--data_dir', default='./data/', type=str, help='Where to find the image data.')
    parser.add_argument('--clean', action='store_true', help='Clean all previously downloaded resources.')
    args = parser.parse_args()

    c = canvas.Canvas(args.output)

    data_dir = Path(args.data_dir)

    tmp_dir = Path('_work')

    if args.clean:
        print('Cleaning previous resources.')
        shutil.rmtree(tmp_dir_path)

    tmp_dir.mkdir(parents=True, exist_ok=True)

    c.setPageSize((400, 700))
    c.drawCentredString(200, 300, 'Proiect 2 Vedere Artificiala')
    c.drawCentredString(300, 500, 'Stavarache Petru-Eric, Grupa 334')
    c.showPage()

    add_task_1_1(c, tmp_dir, data_dir)

    c.save()


if __name__ == '__main__':
    sys.exit(main())

