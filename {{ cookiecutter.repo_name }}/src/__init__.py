import logging
import sys

from .loggable_object import LoggableObject
from .experiment_runner import ExperimentRunner

logging.basicConfig(
    format="%(asctime)s | %(levelname)s : %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)

__all__ = ["LoggableObject", "ExperimentRunner"]
