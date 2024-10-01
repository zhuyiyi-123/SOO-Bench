from soo_bench.protein.exact_oracle import ExactOracle
from soo_bench.protein.discrete_dataset import DiscreteDataset
from soo_bench.protein.tf_bind_8_dataset import TFBind8Dataset
import numpy as np


class TFBind8Oracle(ExactOracle):

    name = "exact_enrichment_score"

    @classmethod
    def supported_datasets(cls):
        return {TFBind8Dataset}

    @classmethod
    def fully_characterized(cls):
        return True

    @classmethod
    def is_simulated(cls):

        return False

    def protected_predict(self, x):

        x_key = tuple(x.tolist())
        return self.sequence_to_score[x_key].astype(np.float32) \
            if x_key in self.sequence_to_score else np.full(
            [1], self.internal_dataset.dataset_min_output, dtype=np.float32)

    def __init__(self, dataset: DiscreteDataset, **kwargs):

        # initialize the oracle using the super class
        super(TFBind8Oracle, self).__init__(
            dataset, is_batched=False,
            internal_batch_size=1, internal_measurements=1,
            expect_normalized_y=False,
            expect_normalized_x=False, expect_logits=False, **kwargs)

        # dictionary containing every point in the search space
        self.sequence_to_score = dict()
        self.internal_dataset._disable_transform = True
        for x, y in self.internal_dataset.iterate_samples():
            self.sequence_to_score[tuple(x.tolist())] = y
        self.internal_dataset._disable_transform = False
