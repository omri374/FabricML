# Iris species detection: experiment template
*This is an experiment template, used to auto-generate notebooks for new experiments*

To generate new notebooks, call `python -m generate_notebook --name my_notebook`

## Experiment description



##### Jupyter helpers:

```python
%reload_ext autoreload
%autoreload 2
```

Define imports

```python
from iris.data import IrisDataLoader
from iris.models import BaseModel
from iris.data_processing import DataProcessor
from iris.experimentation import MlflowExperimentation
from iris.evaluation import IrisEvaluator, IrisEvaluationMetrics
from iris import ExperimentRunner
```

## Load data

```python
data_loader = IrisDataLoader(dataset_name = 'iris', dataset_version = "1")
data_loader.prep_dataset_for_modeling()
data_loader.download_dataset()
X_train, y_train, X_test, y_test = data_loader.get_dataset()

X_train.head()
```

Define experimentation object, which will be used for logging the experiments parameters, metrics and artifacts
*Replace MlflowExperimentation if you use a different experimentation system*
```python
experimentation = MlflowExperimentation()
```

## Preprocessing, modeling and post processing logic
```python
class MyPreprocessor(DataProcessor):
    def apply(self, X):
        return X
    
    def apply_batch(self, X):
        return X

preprocessor = MyPreprocessor()

```

Create model/logic:
```python
class MyModel(BaseModel):
    def fit(self, X, y=None) -> None:
        pass

    def predict(self, X):
        pass

my_model = MyModel(preprocessor = preprocessor)
```

Define evaluation
```python
evaluator = IrisEvaluator()
```


## Run experiment

```python
experiment_runner = ExperimentRunner(
    model=model,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    data_loader=data_loader,
    log_experiment=True,
    experiment_logger=experimentation,
    evaluator=evaluator,
    experiment_name="Experiment",
)

results = experiment_runner.run()
print(results)

```

```python

```
