# Artificial Vision project 1

## Tutorial
Download `Project1.zip` and unzip it, either in the project_root (so that the defaults will work), or in another place, but specifying `--collection_dir` whenever you run a program.
Download the cifar python dataset from https://www.cs.toronto.edu/~kriz/cifar.html and, once again, unzip it in the project root or in your place of choice, specfiying `--cifar_dir`.


## Components

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

`generate_pdf.py` is used to make the project pdf.

Usage example:

`python3 generate_pdf.py out.pdf`

```bash
usage: generate_pdf.py [-h] [--target_dir TARGET_DIR]
                       [--collection_dir COLLECTION_DIR]
                       [--cifar_dir CIFAR_DIR]
                       output

Mimic a picture by creating a grid made out of sample pictures.

positional arguments:
  output                Where to store the generated pdf

optional arguments:
  -h, --help            show this help message and exit
  --target_dir TARGET_DIR
                        Where to find the target images
  --collection_dir COLLECTION_DIR
                        Where to find the collection images
  --cifar_dir CIFAR_DIR
                        Where to find the cifar images
```
