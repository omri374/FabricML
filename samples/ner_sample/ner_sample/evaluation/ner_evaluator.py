from seqeval.metrics import f1_score, accuracy_score

from ner_sample.evaluation import Evaluator, NEREvaluationMetrics


class NEREvaluator(Evaluator):
    """
    This class holds the logic for evaluating a prediction outcome
    y_test in our case is None
    """

    def evaluate(self, y_test, predictions) -> NEREvaluationMetrics:

        golds = []
        predicted = []
        for sentence in predictions:
            gold_tags = [token.get_tag("gold_ner").value for token in sentence.tokens]
            golds.append(gold_tags)
            predicted_tags = [token.get_tag("ner").value for token in sentence.tokens]
            predicted.append(predicted_tags)

        f1 = f1_score(golds, predicted)
        accuracy = accuracy_score(golds, predicted)
        return NEREvaluationMetrics(f1=f1, accuracy=accuracy)
