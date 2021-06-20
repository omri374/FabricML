from iris.evaluation import Evaluator
from iris.evaluation.iris_evaluation_metrics import IrisEvaluationMetrics
from sklearn import metrics


class IrisEvaluator(Evaluator):
    def evaluate(self, y_test, prediction) -> IrisEvaluationMetrics:
        return IrisEvaluationMetrics(
            accuracy=metrics.accuracy_score(prediction, y_test)
        )
