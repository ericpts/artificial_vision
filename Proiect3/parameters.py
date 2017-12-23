class Parameters(object):

    def __init__(self,
                 texture_block_count: int,
                 init_output_width: int,
                 init_output_height: int,
                 block_width: int,
                 block_height: int,
                 nchannels: int,
                 overlap: float = None,
                 transfer_coefficient: float = None,
                 transfer_niterations: int = None):

        self.texture_block_count = texture_block_count

        self.init_output_width = init_output_width
        self.init_output_height = init_output_height

        self.block_width = block_width
        self.block_height = block_height

        self.nchannels = nchannels
        self.overlap = overlap
        self.transfer_coefficient = transfer_coefficient
        self.transfer_niterations = transfer_niterations

        self.overlap_height = int(self.block_height * self.overlap)
        self.overlap_width = int(self.block_width * self.overlap)

        (self.blocks_per_height, self.blocks_per_width) = self.get_blocks_per_dimension()

        # Resize the image to fit the blocks exactly.
        (self.output_width,
         self.output_height) = ((self.blocks_per_height - 1) *
                                (self.block_height - self.overlap_height) + self.block_height,
                                (self.blocks_per_width - 1) *
                                (self.block_width - self.overlap_width) + self.block_width)

    def with_reduce_block_size_by_third(self):
        new_block_width = self.block_width
        new_block_width -= new_block_width // 3

        new_block_height = self.block_height
        new_block_height -= new_block_height // 3

        return Parameters(
            texture_block_count=int(self.texture_block_count * 4 / 3),
            init_output_width=self.init_output_width,
            init_output_height=self.init_output_height,
            block_width=new_block_width,
            block_height=new_block_height,
            nchannels=self.nchannels,
            overlap=self.overlap,
            transfer_coefficient=self.transfer_coefficient,
            transfer_niterations=self.transfer_niterations)

    def get_blocks_per_dimension(self):
        (blocks_per_height,
         blocks_per_width) = (int(1 + (self.init_output_height - self.block_height) //
                                  (self.block_height - self.overlap_height)),
                              int(1 + (self.init_output_width - self.block_width) //
                                  (self.block_width - self.overlap_width)))

        # Sanity checks.
        assert blocks_per_height > 0
        assert blocks_per_width > 0

        # The last blocks should fit completely within the image.
        assert (blocks_per_height - 1) * (self.block_height - self.overlap_height) + \
            self.block_height <= self.init_output_height
        assert (blocks_per_width - 1) * (self.block_width - self.overlap_width) + \
            self.block_width <= self.init_output_width

        # We should not be able to add any more blocks.
        assert (blocks_per_height - 0) * (self.block_height - self.overlap_height) + \
            self.block_height > self.init_output_height
        assert (blocks_per_width - 0) * (
            self.block_width - self.overlap_width) + self.block_width > self.init_output_width

        return (blocks_per_height, blocks_per_width)
