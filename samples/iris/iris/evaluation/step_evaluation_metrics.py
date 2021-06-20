from abc import abstractmethod
from typing import Iterable

from . import EvaluationMetrics


class StepEvaluationMetrics(EvaluationMetrics):
    """
    Class which holds the evaluation output for one model run.
    For example, precision or recall, MSE, accuracy etc.
    StepEvaluationResult assumes that each metric is a list. Some examples:
    1. Metric values per epoch
    2. Metric values per parameter (e.g. threshold)

    This is useful since experiment loggers can log multiple values for each run, which creates a graph.
    See this mlflow example: https://www.mlflow.org/docs/latest/tracking.html#performance-tracking-with-metrics
    """

    @abstractmethod
    def get_metrics(self, step=None):
        """
        Return the evaluation result's metrics you wish to be stored in the experiment logging system
        :param step: Get metric values for a certain step (in case of multiple values,
        like one for each epoch or for each threshold value
        :return: A dictionary with names of values of metrics to store
        """
        pass

    @abstractmethod
    def get_steps(self) -> Iterable:
        """
        Returns an iterable to be used when querying metrics
        """
        pass
