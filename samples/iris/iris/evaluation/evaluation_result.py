from abc import abstractmethod
from typing import Dict

from iris import LoggableObject


class EvaluationResult(LoggableObject):
    """
    Class which holds the evaluation output for one model run.
    For example, precision or recall, MSE, accuracy etc.
    """

    @abstractmethod
    def get_metrics(self) -> Dict:
        """
        Return the evaluation result's metrics you wish to be stored in the experiment logging system
        :return: A dictionary with names of values of metrics to store
        """
        pass

    def get_params(self):
        # Evaluation results are not likely to have params, just metrics
        return None
