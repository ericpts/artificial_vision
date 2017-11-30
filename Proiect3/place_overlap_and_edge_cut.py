from place_utils import *

def place_overlap_and_edge_cut(params: Parameters, blocks: List) -> ndarray:
    """ Returns the generated picture. """

    def start(i: int, j: int):
        (start_height, start_width) = (i * params.block_height, j * params.block_width)
        start_height -= i * overlap_height
        start_width -= j * overlap_width
        return (start_height, start_width)

    def energy_horizontal(i: int, j: int, piece: int) -> ndarray:
        """ Calculate the horizontal energy matrix for placing `piece` on the (i, j)'th spot. """
        assert j != 0 # This is the first piece in its' row, so there is no neighbour to the left.

        overlap_width = int(params.overlap * params.block_width)
        (start_height, start_width) = start(i, j)

        output_chunk = output[start_height : start_height + params.block_height, start_width : start_width + overlap_width]

        # Take all lines of the block, but only columns 0 .. overlap_width - 1 (chop off the left side).
        piece_chunk = blocks[piece][:, : overlap_width]

        energy = np.sum(
                (output_chunk - piece_chunk) ** 2,
                axis=(2, )
                )
        return energy

    def energy_vertical(i: int, j: int, piece: int) -> ndarray:
        """ Calculate the vertical energy matrix for placing `piece` on the (i, j)'th spot. """
        assert i != 0 # This piece is on the first row, so there are no neighbours below it.

        overlap_height = int(params.overlap * params.block_height)
        (start_height, start_width) = start(i, j)

        output_chunk = output[start_height : start_height + overlap_height, start_width : start_width + params.block_width]
        # Take only the top 0..overlap_height - 1 rows, and all columns inside them (chop off the bottom side).
        piece_chunk = blocks[piece][: overlap_height, :]

        energy = np.sum(
                (output_chunk - piece_chunk) ** 2,
                axis=(2, )
                )
        return energy

    def distance_horizontal(i: int, j: int, piece: int) -> float:
        """ Calculates the horizontal overlap cost of placing `piece` on poisition (i, j). """

        energy = energy_horizontal(i, j, piece)
        path = get_column_path_dynamicprogramming(energy)

        ret = 0
        for (i, j) in path.items():
            ret += energy[i, j]
        return ret

    def distance_vertical(i: int, j: int, piece: int) -> float:
        """ Calculates the vertical overlap cost of placing `piece` on poisition (i, j). """

        energy = energy_vertical(i, j, piece)
        path = get_column_path_dynamicprogramming(transpose(energy))

        ret = 0
        for (i, j) in path.items():
            ret += energy[j, i]
        return ret

    def get_best_fit(i: int, j: int) -> int:
        """ Get the index of the best piece to place on (i, j). """
        def cost(piece: int) -> float:
            ret = 0
            if i > 0:
                ret += distance_vertical(i, j, piece)
            if j > 0:
                ret += distance_horizontal(i, j, piece)
            return ret

        sample_blocks = random.sample(range(len(blocks)), min(1000, len(blocks)))
# Only choose a subset of blocks to consider.
# If we do not do this, the picture will end up consisting of the same 3-4 blocks repeated over and over.
        best, best_cost = (sample_blocks[0], cost(sample_blocks[0]))
        for b in sample_blocks:
            now_cost = cost(b)
            if now_cost < best_cost:
                (best, best_cost) = (b, now_cost)

        return best

    overlap_width = int(params.block_width * params.overlap)
    overlap_height = int(params.block_height * params.overlap)

    output = ndarray(shape=(params.output_height - (params.blocks_per_height - 1) * overlap_height
                          , params.output_width - (params.blocks_per_width - 1) * overlap_width
                          , params.nchannels), dtype=np.uint8)

    for i in tqdm(range(params.blocks_per_width)):
        for j in range(params.blocks_per_height):
            piece = get_best_fit(i, j)
            (start_height, start_width) = start(i, j)

# place_mask[i, j] represents whether to use `piece`'s pixel at relative offset (i, j).
            place_mask = np.ones(shape=(params.block_height, params.block_width, 3), dtype=bool)

            if j > 0:
                horizontal_path = get_column_path_dynamicprogramming(energy_horizontal(i, j, piece))
                for (row, col) in horizontal_path.items():
                    place_mask[row, : col] = False
            if i > 0:
                vertical_path = get_column_path_dynamicprogramming(transpose(energy_vertical(i, j, piece)))
                for (col, row) in vertical_path.items():
                    place_mask[: row, col] = False

            np.copyto(dst=output[start_height : start_height + params.block_height,
                                 start_width : start_width + params.block_width],
                      src=blocks[piece],
                      where=place_mask)


    return output
