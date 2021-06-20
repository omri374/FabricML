from typing import Dict

from ner_sample.data_processing import DataProcessor


class EmptyProcessor(DataProcessor):
    """
    Data processor that doesn't do anything to the data
    """

    def apply(self, X):
        return X

    def apply_batch(self, X):
        return X

    def get_params(self) -> Dict:
        return {"processor_name": self.name}
