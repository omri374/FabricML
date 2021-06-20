import os
from pathlib import Path

import pytest

from ner_sample import ExperimentRunner
from ner_sample.evaluation import NEREvaluator
from tests.mocks import MockDataLoader, MockModel, MockExperimentation


def test_experiment_runner():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    mock_data_path = Path(dir_path, "resources").resolve()
    data_loader = MockDataLoader(local_data_path=mock_data_path)
    corpus, test = data_loader.get_dataset()
    train = corpus.train
    assert train is not None
    assert test is not None

    model = MockModel(model_name="Mock", param1="hello", param2="world")
    evaluator = NEREvaluator()
    experiment_logger = MockExperimentation()
    experiment_runner = ExperimentRunner(
        model=model,
        X_train=train,
        X_test=test,
        data_loader=data_loader,
        evaluator=evaluator,
        experiment_logger=experiment_logger,
        experiment_name="Text",
        one_additional_param="Will this work"
    )
    results = experiment_runner.run()

    assert results.f1 == pytest.approx(0.7, 0.1)
    assert results.accuracy == pytest.approx(0.9, 0.1)

    # assert params and metrics are logged
    assert 'param1' in experiment_logger.params
    assert 'param2' in experiment_logger.params

    assert experiment_logger.params['param1'] == 'hello'
    assert experiment_logger.params['param2'] == 'world'
    assert experiment_logger.params['one_additional_param'] == 'Will this work'
    assert experiment_logger.metrics['f1'] == results.f1
    assert experiment_logger.metrics['accuracy'] == results.accuracy
