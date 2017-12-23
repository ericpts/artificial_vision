from place_utils import *


def place_random(params: Parameters, sample_img: ndarray) -> ndarray:
  """ Returns the generated picture. """

  def start(i: int, j: int):
    return start_with_params(params, i, j)

  blocks = generate_blocks(params.texture_block_count, params.block_height, params.block_width,
                           sample_img)

  output = ndarray(
      shape=(params.output_height, params.output_width, params.nchannels), dtype=np.uint8)

  for i in tqdm(range(params.blocks_per_height)):
    for j in range(params.blocks_per_width):
      (start_width, start_height) = start(i, j)
      (end_width, end_height) = (start_width + params.block_width,
                                 start_height + params.block_height)

      piece = random.randint(0, len(blocks) - 1)

      output[start_height:end_height, start_width:end_width] = blocks[piece]

  return output
