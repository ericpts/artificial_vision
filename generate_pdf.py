#!/usr/bin/python3

from os.path import basename
import sys
import argparse
import subprocess
import tempfile
import random
import multiprocessing as mp
from pathlib import Path
from reportlab.pdfgen import canvas
from util import read_image

from lib_google_img.google_images_download import google_images_download

CIFAR_LABELS = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def gen_extra_args():
    ret = []
    desc = 'This picture was generated '

    if random.randint(1, 4) == 1:
        ret.append('--random')
        desc += 'by randomly placing the tiles '
    else:
        desc += 'in a grid '
        if random.randint(1, 4) == 1:
            ret.append('--allow_duplicates')
            desc += 'allowing duplicates '

    ret.append('--horizontal_pieces')
    n = random.choice([25, 50, 75, 100, 200])
    ret.append(str(n))

    desc += ' with {} pieces per horizontal.'.format(n)

    return (ret, desc)

def add_picture_to_canvas(pic, desc, c):
    (w, h) = read_image(pic).shape[0:2]
    c.setPageSize((h, w + 100))
    c.drawInlineImage(str(pic), 0, 0)
    c.drawCentredString(h / 2, w + 50, desc)
    c.showPage()


def add_sample_collection_pictures(c, tmp_dir, collection_dir, targets):
    def gen_for_target(t, extra_args, out_file):
        print('Generating {} ... '.format(out_file))
        gen_args = [str(t), '--collection_dir', collection_dir, str(out_file)] + extra_args
        subprocess.check_call(['python3', 'main.py'] + gen_args)
        print('done generating {}'.format(out_file))

    for t in targets:
        out_file = Path(tmp_dir.name) / Path(t).name
        (extra_args, desc) = gen_extra_args()

        gen_for_target(t, extra_args, out_file)
        add_picture_to_canvas(out_file, desc, c)

def add_cifar_pictures(c, tmp_dir, cifar_dir, cifar_labels):

    def gen_for_label(l, extra_args, out_file):
        """ Returns the url used. """
        print('Generating pic for cif label {} ...'.format(l))

        LIMIT = 10
        links = google_images_download.get_image_links(search_keywords=[l], keywords=['high resolution'], requests_delay=0, limit=LIMIT)

        at = 0
        while at < LIMIT:
            target = links[at]
            try:
                print("Generating {} from {} ... ".format(out_file, target))
                gen_args = [str(target), '--cifar_dir', cifar_dir, '--cifar_label', l, out_file] + extra_args
                subprocess.check_call(['python3', 'main.py'] + gen_args)
                print("done generating {}".format(out_file))
                return target
            except subprocess.CalledProcessError as e:
                print("encountered error in {} for {}, trying next url".format(target, out_file))
            at += 1
        raise Error("Could not generate pic for cifar label {}, increase LIMIT".format(l))

    for l in cifar_labels:
        (extra_args, desc) = gen_extra_args()
        out_file = Path(tmp_dir.name) / (l + '.jpg')

        url = gen_for_label(l, extra_args, out_file)
        desc = 'Cifar picture for label {} from {}: '.format(l, url) + desc
        add_picture_to_canvas(out_file, desc, c)



def main():
    parser = argparse.ArgumentParser(description = 'Mimic a picture by creating a grid made out of sample pictures.')
    parser.add_argument('output', type=str, help='Where to store the generated pdf')
    parser.add_argument('--target_dir', default='./data/imaginiTest/', dest='target_dir', type=str, help='Where to find the target images')
    parser.add_argument('--collection_dir', default='./data/colectie/', dest='collection_dir', type=str, help='Where to find the collection images')
    parser.add_argument('--cifar_dir', default='./cifar-10-batches-py/', dest='cifar_dir', type=str, help='Where to find the cifar images')

    args = parser.parse_args()

    target_path = Path(args.target_dir)
    c = canvas.Canvas(args.output)

    tmp_dir = tempfile.TemporaryDirectory()

    c.setPageSize((400, 700))
    c.drawCentredString(200, 300, 'Proiect 1 Vedere Artificiala')
    c.drawCentredString(300, 500, 'Stavarache Petru-Eric, Grupa 334')
    c.showPage()

    add_sample_collection_pictures(c, tmp_dir, args.collection_dir, target_path.iterdir())
    add_cifar_pictures(c, tmp_dir, args.cifar_dir, CIFAR_LABELS)

    c.save()
if __name__ == '__main__':
    sys.exit(main())
