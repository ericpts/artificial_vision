class Parameters(object):
    def __init__(self,
            output_width: int, output_height: int,
            block_width: int, block_height: int,
            blocks_per_width: int, blocks_per_height: int,
            nchannels: int,
            overlap: float = None):

        self.output_width = output_width
        self.output_height = output_height
        self.block_width = block_width
        self.block_height = block_height
        self.blocks_per_width = blocks_per_width
        self.blocks_per_height = blocks_per_height

        self.nchannels = nchannels
        self.overlap = overlap
