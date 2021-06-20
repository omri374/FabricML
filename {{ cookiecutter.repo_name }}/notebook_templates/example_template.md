# Example experiment template

## Description

This notebook template shows mock implementations of mlfabric's building blocks.
The notebook template file (in markdown) is located in [../notebook_templates/example_template.md]()

To generate a Jupyter notebook from this template, run:

```
python generate_notebook.py --name my_new_notebook.ipynb --template_file example_template.md
```

### Jupyter helpers

```python
%reload_ext autoreload
%autoreload 2
```

### Define imports

```python
from typing import Dict

from src.data import DataLoader
from src.data_processing import DataProcessor, EmptyProcessor
from src.experimentation import Experimentation, MlflowExperimentation
from src.models import BaseModel
from src.evaluation import Evaluator, EvaluationMetrics
from src import ExperimentRunner

```

### Load data using the DataLoader class

```python
DATASET_NAME = "X"
DATASET_VERSION = "0.1"

class MockDataLoader(DataLoader):
    def __init__(self, dataset_name, dataset_version):
        # Mock values, in reality the dataset would be read from file / stream
        self.X_train = [1, 2, 3, 4, 5]
        self.y_train = [1, 1, 1, 0, 0]
        self.X_test = [1, 2, 3, 4, 4]
        self.y_test = [1, 1, 1, 1, 1]

        super().__init__(dataset_name=dataset_name, dataset_version=dataset_version)

    def download_dataset(self) -> None:
        pass

    def get_dataset(self):
        if self.dataset_name == "X" and self.dataset_version == "0.1": 
            return {
            "X_train": self.X_train,
            "y_train": self.y_train,
            "X_test": self.X_test,
            "y_test": self.y_test
            }
        else:
            print(f"Dataset {self.dataset_name} with version {self.dataset_version} not found")

data_loader = MockDataLoader(dataset_name=DATASET_NAME, dataset_version=DATASET_VERSION)
dataset_for_modeling = data_loader.get_dataset()
X_train, y_train = dataset_for_modeling['X_train'], dataset_for_modeling['y_train']
X_test, y_test = dataset_for_modeling['X_test'], dataset_for_modeling['y_test']

```

### Define experimentation logger (which logs params, metrics and code)

```python
experiment_logger = MlflowExperimentation()
```

### Create preprocessor / use existing

```python
# EmptyProcessor does no processing, just returns the input it received
preprocessor = EmptyProcessor()
```

### Define new model/logic

```python
class MockModel(BaseModel):
    def get_params(self) -> Dict:
        return {"param_value": "1"}

    def __init__(self, model_name=None, **hyper_params):
        self.x = None
        super().__init__(model_name=model_name, hyper_params=hyper_params)

    def fit(self, X, y=None) -> None:
        self.x = X

    def predict(self, X):
        return self.x == X


mock_model = MockModel(preprocessor = preprocessor, experiment_logger = experiment_logger)
```

### Define evaluation

Evaluator (how to measure) and EvaluatorMetrics (actual values)

```python
class MockEvaluationMetrics(EvaluationMetrics):
    """
    Class to hold the actual values the evaluation created, e.g. precision, recall, MSE.
    """
    def __init__(self, precision, recall):
        self.precision = precision
        self.recall = recall
        super().__init__()

    def get_metrics(self):
        return {"precision": self.precision, "recall": self.recall}
    
    def __repr__(self):
        return f"Recall: {self.recall}, precision: {self.precision}"


class MockEvaluator(Evaluator):
    """
    Class to hold the logic for how the model is evaluated.
    """
    def __init__(self, expected_recall, expected_precision):
        
        # Mock values. Real values are calculated in the evaluate method
        self.expected_recall = expected_recall
        self.expected_precision = expected_precision
        super().__init__()

    def evaluate(self, predicted, actual) -> MockEvaluationMetrics:
        # This is where actual evaluation takes place. In the mock case it just returns constant values
        return MockEvaluationMetrics(
            recall=self.expected_recall, precision=self.expected_precision
        )


evaluator = MockEvaluator(expected_recall=0.5, expected_precision=0.7)
```

### Run experiment

The ExperimentRunner constructor logs params and metrics from all objects
The `.evaluate()` method runs model prediction, evaluates the results and logs everything as metrics.

```python
experiment_runner = ExperimentRunner(
    model=mock_model,
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    data_loader=data_loader,
    evaluator=evaluator,
    log_experiment=True,
    experiment_logger=experiment_logger,
    experiment_name="Text",
)

# Option 1: Run full experiment (fit, predict, evaluate, log)
results = experiment_runner.run()

# Option 2: Run step by step (this would re-run everything)
experiment_runner.fit_model()
experiment_runner.predict()
results = experiment_runner.evaluate()

# Get intermediate values, after the experiment ended
predictions = experiment_runner.get_predictions()
results = experiment_runner.get_evaluation_metrics()
model = experiment_runner.model

print(results)
```

### Reproducing experiment

Note that Mlflow created a folder called mlruns in this notebook's folder. 
It contains all the values generated during this run. 
Mlflow and other experimentation mechanisms also offer a UI for tracking an experiment. 
If you're using Mlflow, you can also call `mlflow ui` to log into a local mlflow server