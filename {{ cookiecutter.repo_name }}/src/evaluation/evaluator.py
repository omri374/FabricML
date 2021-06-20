from abc import abstractmethod
from typing import Dict

from src import LoggableObject
from . import EvaluationMetrics


class Evaluator(LoggableObject):
    def __init__(self, evaluator_name=None):
        """
        Holds the logic for evaluating model results
        :param evaluator_name Name of evaluator
        """

        super().__init__(name=evaluator_name)

    @abstractmethod
    def evaluate(self, **kwargs) -> EvaluationMetrics:
        """
        Method for running evaluations on model predictions
        :param kwargs: for example: actual, predicted (or other inputs required for evaluation)
        :return: EvaluationMetrics
        """
        pass

    def __repr__(self):
        return f"Evaluator: {self.name}"

    def get_params(self) -> Dict:
        # Evaluators are not likely to have params to log, override if it is different in your case
        pass

    def get_metrics(self) -> Dict:
        # Evaluators are not likely to have metrics to log,
        # override if it is different in your case
        # Metrics themselves should be logged using an EvaluationResult object
        pass
