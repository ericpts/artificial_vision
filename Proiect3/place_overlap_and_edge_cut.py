from place_utils import *


def energy_horizontal_with_params(
        params: Parameters, blocks: List, output: ndarray, i: int, j: int, piece: int) -> ndarray:
    """ Calculate the horizontal energy matrix for placing `piece` on the (i, j)'th spot. """
    assert j != 0  # This is the first piece in its' row, so there is no neighbour to the left.

    overlap_width = int(params.overlap * params.block_width)
    (start_height, start_width) = start_with_params(params, i, j)

    output_chunk = output[start_height: start_height + params.block_height,
                          start_width: start_width + overlap_width]

    # Take all lines of the block, but only columns 0 .. overlap_width - 1 (chop off the left side).
    piece_chunk = blocks[piece][:, : overlap_width]

    return distance_matrix(output_chunk, piece_chunk)


def energy_vertical_with_params(
        params: Parameters, blocks: List, output: ndarray, i: int, j: int, piece: int) -> ndarray:
    """ Calculate the vertical energy matrix for placing `piece` on the (i, j)'th spot. """
    assert i != 0  # This piece is on the first row, so there are no neighbours below it.

    overlap_height = int(params.overlap * params.block_height)
    (start_height, start_width) = start_with_params(params, i, j)

    output_chunk = output[start_height: start_height + overlap_height,
                          start_width: start_width + params.block_width]
    # Take only the top 0..overlap_height - 1 rows, and all columns inside
    # them (chop off the bottom side).
    piece_chunk = blocks[piece][: overlap_height, :]

    return distance_matrix(output_chunk, piece_chunk)


def distance_horizontal_with_params(
        params: Parameters, blocks: List, output: ndarray, i: int, j: int, piece: int) -> ndarray:
    """ Calculates the horizontal overlap cost of placing `piece` on poisition (i, j). """
    return np.sum(energy_horizontal_with_params(params, blocks, output, i, j, piece))


def distance_vertical_with_params(
        params: Parameters, blocks: List, output: ndarray, i: int, j: int, piece: int) -> ndarray:
    """ Calculates the vertical overlap cost of placing `piece` on poisition (i, j). """
    return np.sum(energy_vertical_with_params(params, blocks, output, i, j, piece))


def texture_cost_with_params(
        params: Parameters, blocks: List, output: ndarray, i: int, j: int, piece: int) -> float:
    """ Get the texture-based cost of placing `piece` on (i, j). """
    ret = 0
    if i > 0:
        ret += distance_vertical_with_params(params, blocks, output, i, j, piece)
    if j > 0:
        ret += distance_horizontal_with_params(params, blocks, output, i, j, piece)
    return ret


def place_with_params(
        params: Parameters, blocks: List, output: ndarray, i: int, j: int, piece: int):
    """ We have decided to place `piece` on (i, j).
    Figure out where it should be cut and place it accordingly.
    """
    (start_height, start_width) = start_with_params(params, i, j)

# place_mask[i, j] represents whether to use `piece`'s pixel at relative offset (i, j).
    place_mask = np.ones(shape=(params.block_height, params.block_width, 3), dtype=bool)

    if j > 0:
        horizontal_path = get_column_path_dynamicprogramming(
            energy_horizontal_with_params(params, blocks, output, i, j, piece))
        for (row, col) in horizontal_path.items():
            place_mask[row, : col] = False
    if i > 0:
        vertical_path = get_column_path_dynamicprogramming(
            transpose(energy_vertical_with_params(params, blocks, output, i, j, piece)))
        for (col, row) in vertical_path.items():
            place_mask[: row, col] = False

    np.copyto(dst=output[start_height: start_height + params.block_height,
                         start_width: start_width + params.block_width],
              src=blocks[piece],
              where=place_mask)


def place_overlap_and_edge_cut(params: Parameters, sample_img: ndarray) -> ndarray:
    """ Returns the generated picture. """

    def start(i: int, j: int):
        return start_with_params(params, i, j)

    def energy_horizontal(i: int, j: int, piece: int) -> ndarray:
        return energy_vertical_with_params(params, blocks, output, i, j, piece)

    def energy_vertical(i: int, j: int, piece: int) -> ndarray:
        return energy_horizontal_with_params(params, blocks, output, i, j, piece)

    def distance_horizontal(i: int, j: int, piece: int) -> float:
        return distance_horizontal_with_params(params, blocks, output, i, j, piece)

    def distance_vertical(i: int, j: int, piece: int) -> float:
        return distance_vertical_with_params(params, blocks, output, i, j, piece)

    def get_best_fit(i: int, j: int) -> int:
        """ Get the index of the best piece to place on (i, j). """
        def cost(piece: int) -> float:
            return texture_cost_with_params(params, blocks, output, i, j, piece)

        sample_blocks = random.sample(range(len(blocks)), min(1000, len(blocks)))
# Only choose a subset of blocks to consider.
# If we do not do this, the picture will end up consisting of the same 3-4
# blocks repeated over and over.
        best, best_cost = (sample_blocks[0], cost(sample_blocks[0]))
        for b in sample_blocks:
            now_cost = cost(b)
            if now_cost < best_cost:
                (best, best_cost) = (b, now_cost)

        return best

    def place(i: int, j: int, piece: int):
        place_with_params(params, blocks, output, i, j, piece)

    blocks = generate_blocks(
        params.texture_block_count,
        params.block_height,
        params.block_width,
        sample_img)

    output = ndarray(
        shape=(
            params.output_height,
            params.output_width,
            params.nchannels),
        dtype=np.uint8)

    for i in tqdm(range(params.blocks_per_height)):
        for j in range(params.blocks_per_width):
            piece = get_best_fit(i, j)
            place(i, j, piece)

    return output
