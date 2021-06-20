from abc import abstractmethod

from iris import LoggableObject


class EvaluationMetrics(LoggableObject):
    """
    Class which holds the evaluation output for one model run.
    For example, precision or recall, MSE, accuracy etc.
    """

    @abstractmethod
    def get_metrics(self):
        """
        Return the evaluation result's metrics you wish to be stored in the experiment logging system
        like one for each epoch or for each threshold value
        :return: A dictionary with names of values of metrics to store
        """
        pass

    def get_params(self):
        # Evaluation results are not likely to have params, just metrics
        return None
