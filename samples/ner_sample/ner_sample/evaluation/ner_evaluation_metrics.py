from ner_sample.evaluation import EvaluationMetrics


class NEREvaluationMetrics(EvaluationMetrics):
    """
    This class holds the metrics calculated during the experiment run
    """

    def __init__(self, f1, accuracy):
        self.f1 = f1
        self.accuracy = accuracy
        super().__init__()

    def __repr__(self):
        return f"F1 score: {self.f1}, Accuracy score: {self.accuracy}"

    def get_metrics(self):
        """
        Return a dict with f1 and accuracy values
        """
        return { "f1": self.f1, "accuracy":self.accuracy }