#!/usr/bin/python3

from os.path import basename, exists
import sys
import glob
import functools
import shutil
import argparse
import concurrent.futures
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

algorithms = ['random', 'overlap', 'overlap-and-cut']

def add_picture_to_canvas(pic, desc: str, c: canvas):
    (w, h) = read_image(pic).shape[0:2]
    c.setPageSize((max(h, 500), w + 100))
    c.drawInlineImage(str(pic), 0, 0)
    c.drawCentredString(h / 2, w + 50, desc)
    c.showPage()

def out_file(tmp_dir: Path, sample: Path, algo: str):
    return tmp_dir / '{}_{}_out.jpg'.format(sample.stem, algo)

def gen_from_sample(sample: Path, tmp_dir: Path):
    for algo in algorithms:
        out = out_file(tmp_dir, sample, algo)
        subprocess.check_call(
            ['python3', 'main.py',
             '--sample', sample,
             '--output', str(out),
             '--overlap', str(0.16),
             '--texture-block-size', *['50', '50'],
             '--output-size', *['500', '500'],
             '--algorithm', algo,
             '--texture-block-count', str(10000)],
            stdout=subprocess.DEVNULL)

def gen_textured(texture: Path, out: Path, target: Path):
    subprocess.check_call(['python3', 'main.py',
        '--sample', str(texture),
        '--output', str(out),
        '--overlap', str(0.16),
        '--texture-block-size', *['50', '50'],
        '--texture-block-count', str(1000),
        '--algorithm', 'texture-transfer',
        '--transfer-image', str(target),
        '--transfer-niterations', str(5)],
        stdout=subprocess.DEVNULL)

def add_task_1_1(tmp_dir: Path, data_dir: Path):
    canvas_todos = []

    with concurrent.futures.ProcessPoolExecutor() as e:
        for f in glob.glob(str(data_dir / 'img*.png')):
            sample = Path(f)
            e.submit(gen_from_sample, sample, tmp_dir)

            canvas_todos.append(functools.partial(add_picture_to_canvas, sample, '{} original'.format(sample.stem)))

            for algo in algorithms:
                out = out_file(tmp_dir, sample, algo)
                canvas_todos.append(functools.partial(add_picture_to_canvas, out, '{} with algorithm {}'.format(sample.stem, algo)))

    return canvas_todos

def add_task_1_2(tmp_dir: Path, data_dir: Path):

    target = data_dir / 'eminescu.jpg'
    texture = data_dir / 'rice.jpg'
    out = data_dir / 'eminescu_transfer.jpg'

    canvas_todos = []
    canvas_todos.append(functools.partial(add_picture_to_canvas, target, 'eminescu original'))
    canvas_todos.append(functools.partial(add_picture_to_canvas, out, 'eminescu with rice'))

    with concurrent.futures.ProcessPoolExecutor() as e:
        e.submit(gen_textured(texture, out, target))

    return canvas_todos


def main():
    parser = argparse.ArgumentParser(description='Generate project pdf.')
    parser.add_argument('output', type=str, help='Where to store the generated pdf.')
    parser.add_argument(
        '--data_dir',
        default='./data/',
        type=str,
        help='Where to find the image data.')
    parser.add_argument(
        '--clean',
        action='store_true',
        help='Clean all previously downloaded resources.')
    args = parser.parse_args()

    c = canvas.Canvas(args.output)

    data_dir = Path(args.data_dir)

    tmp_dir = Path('_work')

    if args.clean:
        print('Cleaning previous resources.')
        shutil.rmtree(tmp_dir)

    tmp_dir.mkdir(parents=True, exist_ok=True)

    c.setPageSize((400, 700))
    c.drawCentredString(200, 300, 'Proiect 2 Vedere Artificiala')
    c.drawCentredString(300, 500, 'Stavarache Petru-Eric, Grupa 334')
    c.showPage()

    canvas_todos = []
    with concurrent.futures.ProcessPoolExecutor() as e:
        futures = []
        futures.append(e.submit(add_task_1_1, tmp_dir, data_dir))
        futures.append(e.submit(add_task_1_2, tmp_dir, data_dir))

        for f in futures:
            canvas_todos.extend(f.result())

    for f in canvas_todos:
        f(c)

    c.save()


if __name__ == '__main__':
    sys.exit(main())
