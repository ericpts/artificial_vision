# Artificial Vision project 1

Generate jig-saw puzzles - construct big images from a collection of smaller ones.


## Components

### main

`main.py` is the main program, it generates the collage.
Usage example:

`python3 main.py data/imaginiTest/ferrari.jpeg out.jpeg --use_cifar --cifar_label  automobile`

`python3 main.py https://wallpapercave.com/wp/Serf6yS.jpg out.jpeg`

```bash
usage: main.py [-h] [--collection_dir COLLECTION_DIR] [--cifar_dir CIFAR_DIR]
               [--random] [--allow_duplicates]
               [--horizontal_pieces HORIZONTAL_PIECES] [--use_cifar]
               [--cifar_label CIFAR_LABEL]
               target output

Mimic a picture by creating a grid made out of sample pictures.

positional arguments:
  target                Target picture to recreate
  output                Where to store the generated picture

optional arguments:
  -h, --help            show this help message and exit
  --collection_dir COLLECTION_DIR
                        Where to find the project base pictures
  --cifar_dir CIFAR_DIR
                        Where to find the project CIFAR pictures
  --random              Whether to place pictures randomly
  --allow_duplicates    Whether to allow adjacent duplicates
  --horizontal_pieces HORIZONTAL_PIECES
                        How many pieces to plaze horizontally
  --use_cifar           Whether to use the cifar pictures for construction
  --cifar_label CIFAR_LABEL
                        What type of cifar pictures to use
```

### generate_pdf

`generate_pdf.py` is used to make the project pdf.

Usage example:

`python3 generate_pdf.py out.pdf`

```bash
usage: generate_pdf.py [-h] [--target_dir TARGET_DIR]
                       [--collection_dir COLLECTION_DIR]
                       [--cifar_dir CIFAR_DIR]
                       output

Generate project pdf.

positional arguments:
  output                Where to store the generated pdf

optional arguments:
  -h, --help            show this help message and exit
  --target_dir TARGET_DIR
                        Where to find the target images for the base project
  --collection_dir COLLECTION_DIR
                        Where to find the collection images for the base
                        project
  --cifar_dir CIFAR_DIR
                        Where to find the cifar images
```
