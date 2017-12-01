# Artificial Vision project 3

Texture synthesis - quilting and transfer, based on the
"Image Quilting for Texture Synthesis and Transfer" paper by Efros and Freeman.

## Components

### main

`main.py` is the main program, it generates the collage.
Usage example:

`python3 main.py --sample data/rice.jpg --output dat.png --overlap 0.16 --texture-block-size 50 50 --texture-block-count 1000 --algorithm texture-transfer --transfer-image data/eminescu.jpg  --transfer-niterations 5    `

`python3 main.py --sample data/radishes.jpg --output dat.png --overlap 0.16 --texture-block-size 70 70 --output-size 700 700 --texture-block-count 1000 --algorithm overlap-and-cut
`

```bash
usage: main.py [-h] --sample SAMPLE --output OUTPUT --texture-block-size width
               height --texture-block-count TEXTURE_BLOCK_COUNT
               [--output-size height width] [--overlap OVERLAP] --algorithm
               {random,overlap,overlap-and-cut,texture-transfer}
               [--transfer-image TRANSFER_IMAGE]
               [--transfer-coefficient alpha] [--transfer-niterations niters]

Texture generator.

optional arguments:
  -h, --help            show this help message and exit
  --sample SAMPLE       Texture sample.
  --output OUTPUT       Where to save the generated picture.
  --texture-block-size width height
                        From the texture sample, we will extract blocks of
                        this size.
  --texture-block-count TEXTURE_BLOCK_COUNT
                        How many texture blocks to sample.
  --output-size height width
                        Size of the output picture.
  --overlap OVERLAP     How much neighbouring textures should overlap.
  --algorithm {random,overlap,overlap-and-cut,texture-transfer}
                        Which algorithm to use.
  --transfer-image TRANSFER_IMAGE
                        Which picture to transfer texture onto.
  --transfer-coefficient alpha
                        Alpha weight will be given to texture fitness and (1 -
                        alpha) to correspondence fitness.
  --transfer-niterations niters
                        How many iterations to perform when transferring
                        texture.
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
