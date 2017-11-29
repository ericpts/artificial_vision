#!/usr/bin/python3

from os.path import basename, exists
import sys
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

def add_picture_to_canvas(pic, desc, c):
    (w, h) = read_image(pic).shape[0:2]
    c.setPageSize((max(h, 500), w + 100))
    c.drawInlineImage(str(pic), 0, 0)
    c.drawCentredString(h / 2, w + 50, desc)
    c.showPage()


def add_task_1_1(c, tmp_dir_path: Path, data_dir):
    out = tmp_dir_path / 'task_1_1.jpg'
    subprocess.check_call(
            ['python3', 'main.py',
                '--sample', ])

def main():
    parser = argparse.ArgumentParser(description = 'Generate project pdf.')
    parser.add_argument('output', type=str, help='Where to store the generated pdf.')
    parser.add_argument('--data_dir', default='./data/', type=str, help='Where to find the image data.')
    args = parser.parse_args()

    c = canvas.Canvas(args.output)

    data_dir = Path(args.data_dir)

    tmp_dir = Path('_work')
    # tmp_dir = tempfile.TemporaryDirectory()
    tmp_dir_path = Path(tmp_dir.name)

    c.setPageSize((400, 700))
    c.drawCentredString(200, 300, 'Proiect 2 Vedere Artificiala')
    c.drawCentredString(300, 500, 'Stavarache Petru-Eric, Grupa 334')
    c.showPage()

    add_task_1_1(c, tmp_dir_path, data_dir)

    c.save()


if __name__ == '__main__':
    sys.exit(main())

