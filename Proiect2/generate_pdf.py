#!/usr/bin/python3

from os.path import basename, exists
import sys
import argparse
import subprocess
import tempfile
import random
import multiprocessing as mp
from pathlib import Path
from reportlab.pdfgen import canvas

sys.path.append('../')
from util import read_image

from lib_google_img.google_images_download import google_images_download

def add_picture_to_canvas(pic, desc, c):
    (w, h) = read_image(pic).shape[0:2]
    c.setPageSize((h, w + 100))
    c.drawInlineImage(str(pic), 0, 0)
    c.drawCentredString(h / 2, w + 50, desc)
    c.showPage()

def gen_with_args(c, tmp_dir, data_dir, pic_name, desc, args):

    target = str(data_dir / '{}.jpg'.format(pic_name))

    if not exists(target):
        target = str(data_dir / '{}.jpeg'.format(pic_name))

    output = str(tmp_dir / '{}_out.jpg'.format(pic_name))
    subprocess.check_call(
            ['python3', 'main.py',
                target, output,
                *args])
    add_picture_to_canvas(output, desc, c)

def add_task_1_1(c, tmp_dir, data_dir):
    print('Generating task 1.1')
    gen_with_args(c, tmp_dir, data_dir,
            'castel', 'Castel.jpg with -50 columns',
            ['--columns', '-50'])

def add_task_1_2(c, tmp_dir, data_dir):
    print('Generating task 1.2')
    gen_with_args(c, tmp_dir, data_dir,
            'praga', 'Praga.jpg with -100 lines',
            ['--lines', '-100'])

def add_task_1_3(c, tmp_dir, data_dir):
    print('Generating task 1.3')
    gen_with_args(c, tmp_dir, data_dir,
            'delfin', 'Delfin.jpg with +50 lines and +50 columns',
            ['--lines', '50', '--columns', '50'])


def main():
    parser = argparse.ArgumentParser(description = 'Generate project pdf.')
    parser.add_argument('output', type=str, help='Where to store the generated pdf')
    parser.add_argument('--data_dir', default='./data/', type=str, help='Where to find the image data')

    args = parser.parse_args()

    c = canvas.Canvas(args.output)

    data_dir = Path(args.data_dir)

    tmp_dir = tempfile.TemporaryDirectory()
    tmp_dir_path = Path(tmp_dir.name)

    c.setPageSize((400, 700))
    c.drawCentredString(200, 300, 'Proiect 2 Vedere Artificiala')
    c.drawCentredString(300, 500, 'Stavarache Petru-Eric, Grupa 334')
    c.showPage()

    add_task_1_1(c, tmp_dir_path, data_dir)
    add_task_1_2(c, tmp_dir_path, data_dir)
    add_task_1_3(c, tmp_dir_path, data_dir)

    c.save()


if __name__ == '__main__':
    sys.exit(main())

