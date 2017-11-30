from place_utils import *

def place_random(params: Parameters, blocks: List) -> ndarray:
    """ Returns the generated picture. """

    output = ndarray(shape=(params.output_height, params.output_width, params.nchannels), dtype=np.uint8)
    for i in range(params.blocks_per_height):
        for j in range(params.blocks_per_width):
            (start_width, start_height) = (i * params.block_width, j * params.block_height)
            (end_width, end_height) = (start_width + params.block_width, start_height + params.block_height)

            piece = random.randint(0, len(blocks) - 1)

            output[start_height: end_height, start_width: end_width] = blocks[piece]

    return output
