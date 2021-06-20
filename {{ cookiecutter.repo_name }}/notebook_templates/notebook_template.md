# Experiment template
*This is an experiment template, used to auto-generate notebooks for new experiments*

## Experiment description



##### Jupyter helpers:

```python
%reload_ext autoreload
%autoreload 2
```

Define imports

```python

from src.data import DataLoader
from src.models import BaseModel
from src.data_processing import DataProcessor
from src.experimentation import MlflowExperimentation
from src.evaluation import Evaluator, EvaluationMetrics
from src import ExperimentRunner

```

## Load data
*replace MyDataLoader with your DataLoader implementation*

```python
data_loader = MyDataLoader()
data_loader.download_dataset()
dataset_for_modeling = data_loader.get_dataset()
pickle_data = pickle.load(dataset_for_modeling)
X_train, y_train = pickle_data['X_train'], pickle_data['y_train']
X_test, y_test = pickle_data['X_test'], pickle_data['y_test']
```

Define experimentation object, which will be used for logging the experiments parameters, metrics and artifacts
*Replace MlflowExperimentation if you use a different experimentation system*
```python
experimentation = MlflowExperimentation()
``` 

Create preprocessor for handling data preprocessing, feature engineering etc.
```python
class MyPreprocessor(DataProcessor):
    def apply(self, X):
        pass

    def apply_batch(self, X):
        pass

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
class MyEvaluator(Evaluator):
    def evaluate(self, **kwargs) -> EvaluationMetrics:
        pass

evaluator = MyEvaluator()
```


Run experiment

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