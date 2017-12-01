from place_utils import *
from place_overlap_and_edge_cut import *
import pdb


def transfer_texture(
        params: Parameters,
        blocks: List,
        image: ndarray,
        ) -> ndarray:

    def start(i: int, j: int):
        return start_with_params(params, i, j)

    def correspondence_cost(i: int, j: int, piece: int) -> float:
        (start_height, start_width) = start(i, j)
        return np.sum(distance_matrix(
                image[start_height : start_height + params.block_height,
                      start_width : start_width + params.block_width],
                blocks[piece]))

    def get_best_fit(i: int, j: int) -> int:
        """ Get the index of the best piece to place on (i, j). """
        def cost(piece: int) -> float:
            return params.transfer_coefficient * texture_cost_with_params(params, blocks, output, i, j, piece) + (1 - params.transfer_coefficient) * correspondence_cost(i, j, piece)

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
    output = ndarray(shape=image.shape, dtype=image.dtype)

    for i in tqdm(range(params.blocks_per_height)):
        for j in range(params.blocks_per_width):
            piece = get_best_fit(i, j)

            (start_height, start_width) = start(i, j)

            image_chunk = image[start_height : start_height + params.block_height,
                    start_width : start_width + params.block_width]

            # plt.imshow(image_chunk)
            # plt.show()
            # plt.imshow(blocks[piece])
            # plt.show()
            # plt.imshow(distance_matrix(image_chunk, blocks[piece]))
            # plt.show()
            place_with_params(params, blocks, output, i, j, piece)

    return output
