import numbers
from abc import ABC, abstractmethod
from typing import Dict


class LoggableObject(ABC):
    """
    Base abstract class for all objects, enforces the implementation of get_params and get_metrics
    to allow logging of all parts of the pipeline into the experiment logger
    """
    def __init__(self, name=None):
        self.name = name
        if not name:
            self.name = self.__class__.__name__

    def get_params(self) -> Dict:
        """
        Return a dictionary of parameters used by this class, for experiment logging
        :return: a dictionary of all fields which are either numbers, booleans or str.
        """
        fields_to_store = {
            k: v
            for k, v in vars(self).items()
            if isinstance(v, numbers.Number)
            or isinstance(v, bool)
            or isinstance(v, str)
        }

        return fields_to_store

    @abstractmethod
    def get_metrics(self) -> Dict:
        """
        Return a dictionary of metrics used by this class, for experiment logging
        :return:
        """
        pass
