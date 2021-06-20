from abc import abstractmethod
from typing import Dict

from ner_sample import LoggableObject


class DataProcessor(LoggableObject):
    def __init__(self, processor_name=None):
        """
        Umbrella abstract class for all pre or post data processors.
        Inherit from the class to implement data cleaning, preprocessing or postprocessing
        Treats data prior/after to running a model
        :param processor_name Name of processor
        """

        super().__init__(name=processor_name)

    @abstractmethod
    def apply(self, **kwargs):
        pass

    @abstractmethod
    def apply_batch(self, **kwargs):
        pass

    def get_metrics(self) -> Dict:

        # Data processors are not likely to return any metrics, just params
        return None

    def __repr__(self):
        return f"Processor: {self.name}"
