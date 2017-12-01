# Artificial Vision project 2

Smart image resizing based on the
"Seam Carving for Content-Aware Image Resizing" paper by Avidan and Shamir.

## Components

### main

`main.py` is the main program, it generates the collage.
Usage example:

`python3 main.py data/delfin.jpeg delfin.jpg --columns 50 --lines 50 `

`python3 main.py data/lac.jpg delfin.jpg --remove-object-mouse  `

```bash
usage: main.py [-h] [--columns COLUMNS] [--lines LINES]
               [--path-algorithm {random,greedy,dynamicprogramming}]
               [--remove-object-fixed-coordinates x0 y0 x1 y1]
               [--remove-object-mouse]
               target output

Smartly resize imges.

positional arguments:
  target                Target picture to resize.
  output                Where to save the picture.

optional arguments:
  -h, --help            show this help message and exit
  --columns COLUMNS     Desired change in columns. Positive means add,
                        negative means remove.
  --lines LINES         Desired change in lines. Positive means add, negative
                        means remove.
  --path-algorithm {random,greedy,dynamicprogramming}
                        Algorithm to use for path selection.
  --remove-object-fixed-coordinates x0 y0 x1 y1
                        The coordinates of the rectangle to remove.
  --remove-object-mouse
                        Select the object to be removed by using the mouse.
```

### generate_pdf

`generate_pdf.py` is used to make the project pdf.

Usage example:

`python3 generate_pdf.py out.pdf`

```bash
usage: generate_pdf.py [-h] [--data_dir DATA_DIR] [--clean] output

Generate project pdf.

positional arguments:
  output               Where to store the generated pdf.

optional arguments:
  -h, --help           show this help message and exit
  --data_dir DATA_DIR  Where to find the image data.
  --clean              Clean all previously downloaded resources.
```
