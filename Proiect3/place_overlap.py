from place_utils import *
import pdb

def distance(x: ndarray, y: ndarray) -> float:
    return np.sum((x - y) ** 2)

def place_overlap(params: Parameters, blocks: List) -> ndarray:
    """ Returns the generated picture. """

    def start(i: int, j: int):
        return start_with_params(params, i, j)

    def place_piece_on_output(output: ndarray, i: int, j: int, piece: int):
        (start_height, start_width) = start(i, j)
        (end_height, end_width) = (start_height + params.block_height, start_width + params.block_width)

        output[start_height: end_height, start_width: end_width] = blocks[piece]

    def cost_horizontal(i: int, j: int, piece: int) -> float:
        """ Calculate the horizontal cost of placing `piece` on the (i, j)'th spot. """
        overlap_width = int(params.overlap * params.block_width)

        (start_height, start_width) = start(i, j)

        if j == 0:
            # This is the first piece in its' row, so there is no neighbour to the left. Therefore, the cost is 0.
            return 0

        return distance(
                    output[start_height : start_height + params.block_height, start_width : start_width + overlap_width],
                    # Take all lines of the block, but only columns 0 .. overlap_width - 1 (chop off the left side).
                    blocks[piece][:, : overlap_width])

    def cost_vertical(i: int, j: int, piece: int) -> float:
        """ Calculate the vertical cost of placing `piece` on the (i, j)'th spot. """
        overlap_height = int(params.overlap * params.block_height)

        (start_height, start_width) = start(i, j)

        if i == 0:
            # This piece is on the first row, so there are no neighbours below it.
            return 0

        return distance(
                    output[start_height : start_height + overlap_height, start_width : start_width + params.block_width],
                    # Take only the top 0..overlap_height - 1 rows, and all columns inside them (chop off the bottom side).
                    blocks[piece][: overlap_height, :]
                )

    def cost_overlap_vertical_horizontal(i: int, j: int, piece: int) -> float:
        """ There is a rectangle where the vertical and the horizontal costs overlap.
            We count it twice, once for vertical and once for horizontal, therefore we have to subtract it.
        """

        if i == 0 or j == 0:
            return 0

        overlap_width = int(params.overlap * params.block_width)
        overlap_height = int(params.overlap * params.block_height)

        (start_height, start_width) = start(i, j)

        return distance(
                    output[start_height : start_height + overlap_height, start_width : start_width + overlap_width],
                    # Take only the bottom 0..overlap_height - 1 rows, and all columns inside them (chop off the bottom side).
                    blocks[piece][: overlap_height, : overlap_width]
                )


    def get_best_fit(i: int, j: int) -> int:
        """ Get the best piece index to place on (i, j). """
        def cost(piece: int) -> float:
            ret = cost_horizontal(i, j, piece) + cost_vertical(i, j, piece) - cost_overlap_vertical_horizontal(i, j, piece)
            return ret

# Only choose a subset of blocks to consider.
# If we do not do this, the picture will end up consisting of the same 3-4 blocks repeated over and over.
        sample_blocks = range(len(blocks))
        best, best_cost = (sample_blocks[0], cost(sample_blocks[0]))
        for b in sample_blocks:
            now_cost = cost(b)
            if now_cost < best_cost:
                (best, best_cost) = (b, now_cost)

        return best

    overlap_height = int(params.block_height * params.overlap)
    overlap_width = int(params.block_width * params.overlap)

    output = ndarray(shape=(params.output_height - (params.blocks_per_height - 1) * overlap_height
                          , params.output_width - (params.blocks_per_width - 1) * overlap_width
                          , params.nchannels), dtype=np.uint8)

    place_piece = functools.partial(place_piece_on_output, output)

    for i in tqdm(range(params.blocks_per_height)):
        for j in range(params.blocks_per_width):
            # plt.figure()
            # plt.imshow(output)
            place_piece(i, j, get_best_fit(i, j))
            # plt.show()

    return output

