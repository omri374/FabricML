import logging
from time import time

from .evaluation_metrics import EvaluationMetrics
from .iris_evaluation_metrics import IrisEvaluationMetrics
from .step_evaluation_metrics import StepEvaluationMetrics
from .evaluator import Evaluator
from .iris_evaluator import IrisEvaluator


class TimeTook(object):
    """
    Calculates the time a block took to run.
    Example usage:
    with TimeTook("sample"):
        s = [x for x in range(10000000)]
    Modified from:
    https://blog.usejournal.com/how-to-create-your-own-timing-context-manager-in-python-a0e944b48cf8
    """

    def __init__(self, description):
        self.description = description

    def __enter__(self):
        self.start = time()

    def __exit__(self, type, value, traceback):
        self.end = time()
        logging.info(f"Time took for {self.description}: {self.end - self.start}")


__all__ = [
    "EvaluationMetrics",
    "Evaluator",
    "StepEvaluationMetrics",
    "IrisEvaluationMetrics",
    "IrisEvaluator",
    "TimeTook",
]
