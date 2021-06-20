# Experiment template

*This is an experiment template, used to auto-generate notebooks for new experiments*

## Experiment description

##### Jupyter helpers

```python
%reload_ext autoreload
%autoreload 2
```

Define imports

```python
from flair.data import Corpus

from ner_sample.data import ConllDataLoader
from ner_sample.experimentation import MlflowExperimentation
from ner_sample.evaluation import NEREvaluator
from ner_sample import ExperimentRunner
```

## Load data
Download (if missing) the Conll-2003 dataset from Github and load it into memory using a flair Corpus object

```python
data_loader = ConllDataLoader(dataset_name = "conll_03")
data_loader.download_dataset()
train_corpus, test = data_loader.get_dataset() #train_corpus is a flair Corpus containing train and dev
train = train_corpus.train
dev = train_corpus.dev

```

Define experimentation object, which will be used for logging the experiments parameters, metrics and artifacts

```python
experimentation = MlflowExperimentation()
```

Model: Implement a new model here (replace MyNERModel with your own)

```python
class MyNERModel(BaseModel):

    def __init__(self, hyper_param1, hyper_param2, ...):
        super().__init__(hyper_param1=hyper_param1, hyper_param2=hyper_param2, ...)

    def fit(self, X, y=None) -> None:
        pass

    def predict(self, X):
        pass

```

```python
model = MyNERModel()

model.fit(corpus=train_corpus) #train_corpus contains train+dev
```

Define evaluation

```python
evaluator = NEREvaluator()
```

Set up experiment

```python
experiment_runner = ExperimentRunner(
    model=model,
    X_train=train,
    X_test=test,
    data_loader=data_loader,
    log_experiment=True,
    experiment_logger=experimentation,
    evaluator=evaluator,
    experiment_name="Experiment"
)

```

Run experiment

```python
results = experiment_runner.evaluate()
print(results)
```
