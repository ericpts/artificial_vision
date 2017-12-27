#!/usr/bin/python3

from os.path import exists
import sys
import shutil
import argparse
import subprocess
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


def gen_output(target, output, imresize_file, imresize_fn, args):
    subprocess.check_call(['python3', 'main.py', target, output, *args])

    misc.imsave(imresize_file, imresize_fn(read_image(target)))


def add_generated_to_canvas(c, tmp_dir, target, output, imresize_file, desc):
    add_picture_to_canvas(target, desc + ': initial picture', c)
    add_picture_to_canvas(imresize_file, desc + ': modified with imresize', c)
    add_picture_to_canvas(output, desc + ': modified with paper algorithm', c)


def add_from_data(c, tmp_dir, target, output, imresize_file, desc, args, imresize_fn):
    gen_output(target, output, imresize_file, imresize_fn, args)
    add_generated_to_canvas(c, tmp_dir, target, output, imresize_file, desc)


def add_from_name(c, tmp_dir, data_dir, pic_name, desc, args, imresize_fn=None):

    target = str(data_dir / '{}.jpg'.format(pic_name))
    if not exists(target):
        target = str(data_dir / '{}.jpeg'.format(pic_name))

    output = str(tmp_dir / '{}_out.jpg'.format(pic_name))
    imresize_file = str(tmp_dir / '{}_imresize.jpg'.format(pic_name))

    add_from_data(c, tmp_dir, target, output, imresize_file, desc, args, imresize_fn)


def delta_imresize(lines, columns):

    def fn(img):
        (n, m) = img.shape[0:2]
        return misc.imresize(img, (n + lines, m + columns))

    return fn


def add_task_1_1(c, tmp_dir, data_dir):
    print('Generating task 1.1')
    add_from_name(c, tmp_dir, data_dir, 'castel', 'Castel.jpg with -50 columns',
                  ['--columns', '-50'], delta_imresize(0, -50))


def add_task_1_2(c, tmp_dir, data_dir):
    print('Generating task 1.2')
    add_from_name(c, tmp_dir, data_dir, 'praga', 'Praga.jpg with -100 lines', ['--lines', '-100'],
                  delta_imresize(-100, 0))


def add_task_1_3(c, tmp_dir, data_dir):
    print('Generating task 1.3')
    add_from_name(c, tmp_dir, data_dir, 'delfin', 'Delfin.jpg with +50 lines and +50 columns',
                  ['--lines', '50', '--columns', '50'], delta_imresize(50, 50))


def add_task_1_4(c, tmp_dir, data_dir):
    print('Generating task 1.4')
    add_from_name(c, tmp_dir, data_dir, 'lac', 'Lac.jpg with the little girl removed',
                  ['--remove-object-fixed-coordinates', *[str(x) for x in [168, 397, 205, 430]]],
                  delta_imresize(-(205 - 168), -(430 - 397)))


def add_task_1_5(c, tmp_dir, data_dir):
    print('Generating task 1.5')

    keywords = [
        'the persistence of memory',
        'guitar',
        'starry night',
        'cappuccino',
        'cote d\'azur',
    ]

    def random_modification(n, m):

        def var(x):

            def n():
                return x - int(random.normalvariate(x, x / 10))

            r = n()
            while r <= -x or r >= x:
                r = n()
            return r

        return (var(n), var(m))

    def random_path_algorithm():
        return random.choice(['random', 'greedy', 'dynamicprogramming'])

    LIMIT = 10

    def process_fn(keyword, target, output, imresize_file, desc_file):

        def ensure_target_exists():
            if Path(target).exists():
                return
            links = google_images_download.get_image_links(
                search_keywords=[keyword],
                keywords=['high resolution'],
                requests_delay=0,
                limit=LIMIT)
            for target_link in links:
                try:
                    print('Trying to retrieve target link {}'.format(target_link))
                    misc.imsave(target, read_image(target_link))
                    break
                except urllib.error.HTTPError:
                    continue
                except OSError:
                    continue

        ensure_target_exists()
        target_img = read_image(target)
        (n, m) = target_img.shape[0:2]
        (del_lines, del_columns) = random_modification(n, m)
        path_algorithm = random_path_algorithm()

        args = [
            '--lines',
            str(del_lines),
            '--columns',
            str(del_columns),
            '--path-algorithm',
            path_algorithm,
        ]

        with open(desc_file, 'w+t') as f:
            f.write('{} from google, with {} lines, {} columns and path algorithm {}'.format(
                keyword, del_lines, del_columns, path_algorithm))

        gen_output(
            target,
            output,
            imresize_file,
            args=args,
            imresize_fn=delta_imresize(del_lines, del_columns))

    workers = []

    worker_files = []
    for (at, keyword) in enumerate(keywords):
        target = str(tmp_dir / '1_5_{}_in.jpg'.format(at))
        output = str(tmp_dir / '1_5_{}_out.jpg'.format(at))
        imresize_file = str(tmp_dir / '1_5_{}_imresize.jpg'.format(at))
        desc_file = str(tmp_dir / '1_5_{}_desc.txt'.format(at))

        workers.append(
            mp.Process(target=process_fn, args=(keyword, target, output, imresize_file, desc_file)))
        worker_files.append((target, output, imresize_file, desc_file))

    for w in workers:
        w.start()

    for w in workers:
        w.join()

    for (target, output, imresize_file, desc_file) in worker_files:
        with open(desc_file, 'r+t') as f:
            add_generated_to_canvas(c, tmp_dir, target, output, imresize_file, f.read())


def main():
    parser = argparse.ArgumentParser(description='Generate project pdf.')
    parser.add_argument('output', type=str, help='Where to store the generated pdf.')
    parser.add_argument(
        '--data_dir', default='./data/', type=str, help='Where to find the image data.')
    parser.add_argument(
        '--clean', action='store_true', help='Clean all previously downloaded resources.')
    args = parser.parse_args()

    c = canvas.Canvas(args.output)

    data_dir = Path(args.data_dir)

    tmp_dir = Path('_work')
    # tmp_dir = tempfile.TemporaryDirectory()
    tmp_dir_path = Path(tmp_dir.name)

    if args.clean:
        print('Cleaning previous resources.')
        shutil.rmtree(tmp_dir_path)
        tmp_dir.mkdir(parents=True, exist_ok=True)

    c.setPageSize((400, 700))
    c.drawCentredString(200, 300, 'Proiect 2 Vedere Artificiala')
    c.drawCentredString(300, 500, 'Stavarache Petru-Eric, Grupa 334')
    c.showPage()

    add_task_1_1(c, tmp_dir_path, data_dir)
    add_task_1_2(c, tmp_dir_path, data_dir)
    add_task_1_3(c, tmp_dir_path, data_dir)
    add_task_1_4(c, tmp_dir_path, data_dir)
    add_task_1_5(c, tmp_dir_path, data_dir)

    c.save()


if __name__ == '__main__':
    sys.exit(main())
