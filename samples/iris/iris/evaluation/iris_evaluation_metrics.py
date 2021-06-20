from iris.evaluation import EvaluationMetrics


class IrisEvaluationMetrics(EvaluationMetrics):
    def __init__(self, accuracy):
        self.accuracy = accuracy

    def get_metrics(self):
        return {"accuracy": self.accuracy}

    def __repr__(self):
        return str(self.__dict__)
