from place_utils import *
from place_overlap_and_edge_cut import *
import pdb


def transfer_texture(
        params: Parameters,
        texture: ndarray,
        image: ndarray,
        ) -> ndarray:

    def start(i: int, j: int):
        return start_with_params(params, i, j)

    def texture_cost(i: int, j: int, piece: int) -> float:
        return texture_cost_with_params(params, blocks, output, i, j, piece) + texture_cost_with_params(params, blocks, previous_output, i, j, piece)

    def get_best_fit(i: int, j: int, alpha: float, blocks: List) -> int:
        """ Get the index of the best piece to place on (i, j). """
        def correspondence_cost(piece: int):
            return np.sum(distance_matrix(
                image_chunk,
                blocks[piece]))
        def cost(piece: int) -> float:
            return alpha * texture_cost(i, j, piece) + (1 - alpha) * correspondence_cost(piece)

        (start_height, start_width) = start(i, j)
        image_chunk = image[start_height : start_height + params.block_height,
                start_width : start_width + params.block_width]

# Only choose a subset of blocks to consider.
# If we do not do this, the picture will end up consisting of the same 3-4
# blocks repeated over and over.
        sample_blocks = random.sample(range(len(blocks)), min(10000, len(blocks)))
        best, best_cost = (sample_blocks[0], cost(sample_blocks[0]))
        for b in sample_blocks:
            now_cost = cost(b)
            if now_cost < best_cost:
                (best, best_cost) = (b, now_cost)
        return best

    def iter(alpha: float, blocks: List):
        for i in tqdm(range(params.blocks_per_height)):
            for j in range(params.blocks_per_width):
                piece = get_best_fit(i, j, alpha, blocks)
                place_with_params(params, blocks, output, i, j, piece)

    def make_blocks(params: Parameters):
        return generate_blocks(params.texture_block_count, params.block_height, params.block_width, texture)

    output = np.zeros(shape=image.shape, dtype=image.dtype)
    previous_output = np.zeros(shape=image.shape, dtype=image.dtype)

    if params.transfer_niterations == 1:
        blocks = make_blocks(params)
        iter(params.transfer_coefficient, blocks)
        return output

    for i in range(params.transfer_niterations):
        blocks = make_blocks(params)
        coef = 0.8 * i / (params.transfer_niterations - 1) + 0.1
        iter(coef, blocks)

        previous_output = output
        output = ndarray(shape=image.shape, dtype=image.dtype)

        params = params.with_reduce_block_size_by_third()

    return previous_output
