from typing import Dict

from src.data import DataLoader
from src.experimentation import Experimentation, MlflowExperimentation
from src.models import BaseModel
from src.evaluation import Evaluator, EvaluationMetrics
from src import ExperimentRunner


class MockModel(BaseModel):
    def get_params(self) -> Dict:
        return {"param_value": "1"}

    def __init__(self, model_name=None, **hyper_params):
        self.x = None
        super().__init__(model_name=model_name, hyper_params=hyper_params)

    def fit(
        self, X, y=None, experimentation: Experimentation = None, **fit_params
    ) -> None:
        self.x = X

    def predict(self, X):
        return [self.x == X]


class MockEvaluationMetrics(EvaluationMetrics):
    def __init__(self, precision, recall):
        self.precision = precision
        self.recall = recall
        super().__init__()

    def get_metrics(self):
        return {"precision": self.precision, "recall": self.recall}


class MockEvaluator(Evaluator):
    def __init__(self, expected_recall, expected_precision):
        self.expected_recall = expected_recall
        self.expected_precision = expected_precision
        super().__init__()

    def evaluate(self, predicted, actual) -> EvaluationMetrics:
        return MockEvaluationMetrics(
            recall=self.expected_recall, precision=self.expected_precision
        )


class MockDataLoader(DataLoader):
    def __init__(
        self, X_train, y_train, X_test, y_test, dataset_name="X", dataset_version=1
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        super().__init__(dataset_name=dataset_name, dataset_version=dataset_version)

    def download_dataset(self) -> None:
        pass

    def get_dataset(self, dataset_name, dataset_version):
        return self.X_train, self.y_train, self.X_test, self.y_test


def test_experiment_runner():
    X_train = [1, 2, 3, 4, 5]
    y_train = [1, 1, 1, 0, 0]
    X_test = [1, 2, 3, 4, 4]
    y_test = [1, 1, 1, 1, 1]

    data_loader = MockDataLoader(
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )

    expected_recall = 0.5
    expected_precision = 0.7

    model = MockModel(model_name="Mock", param1="hello", param2="world")
    evaluator = MockEvaluator(
        expected_recall=expected_recall, expected_precision=expected_precision
    )
    experiment_logger = MlflowExperimentation()
    experiment_runner = ExperimentRunner(
        model=model,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        data_loader=data_loader,
        evaluator=evaluator,
        experiment_logger=experiment_logger,
        experiment_name="Text",
    )
    results = experiment_runner.run()

    assert results.precision == expected_precision
    assert results.recall == expected_recall
