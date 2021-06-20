import copy

import pytest

from ner_sample.evaluation import NEREvaluator
from tests.mocks import MockDataLoader


def test_ner_evaluator_mock_data(dataset_loader: MockDataLoader):
    corpus, test = dataset_loader.get_dataset()

    # replace some tags to validate evaluation
    counter = 0
    for sentence in test:
        for token in sentence.tokens:
            # "predict" the actual labels, and add some noise
            token.annotation_layers["ner"] = copy.deepcopy(token.annotation_layers["gold_ner"])
            if counter % 3 == 0:
                token.annotation_layers["ner"][0].value = 'O'
            counter += 1

    evaluator = NEREvaluator()
    ner_evaluation_metrics = evaluator.evaluate(y_test=None, predictions=test)
    assert ner_evaluation_metrics.f1 == pytest.approx(0.4, 0.1)
    assert ner_evaluation_metrics.accuracy == pytest.approx(0.9, 0.1)
