class Parameters(object):

    def __init__(self, **kwargs):

        for (k, v) in kwargs.items():
            setattr(self, k, v)

        def set_if_missing(name: str, val):
            if name not in kwargs:
                setattr(self, name, val)

        set_if_missing('points_on_height', 10)
        set_if_missing('points_on_weidth', 10)

        set_if_missing('positive_training_dir', 'positive_training')
        set_if_missing('negative_training_dir', 'negative_training')

        set_if_missing('positive_testing_dir', 'positive_testing')
        set_if_missing('negative_testing_dir', 'negative_testing')

        set_if_missing('cells_per_block', 3)
