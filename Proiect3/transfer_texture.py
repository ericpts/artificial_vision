from place_utils import *


def transfer_texture(
        params: Parameters,
        image: ndarray,
        blocks: List) -> ndarray:
    output = ndarray(shape=image.shape, dtype=image.dtype)

    for i in tqdm(range(params.blocks_per_height)):
        for j in range(params.blocks_per_width):

    return output
