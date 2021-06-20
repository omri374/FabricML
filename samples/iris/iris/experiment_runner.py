import logging

from iris.evaluation import StepEvaluationMetrics

from . import LoggableObject
from .data.data_loader import DataLoader
from .evaluation import EvaluationMetrics, Evaluator
from .experimentation import Experimentation
from .models import BaseModel

logger = logging.getLogger(__name__)


class ExperimentRunner:
    def __init__(
        self,
        model: BaseModel,
        X_train,
        X_test,
        data_loader: DataLoader,
        evaluator: Evaluator,
        y_test=None,
        y_train=None,
        log_experiment: bool = True,
        experiment_logger: Experimentation = None,
        experiment_name: str = None,
        **experiment_params_to_log,
    ):
        """
        Runs one model, evaluates results, and stores all parameters
        and metrics in the experimentation service

        :param model: model instance (of type BaseModel)
        :param X_train: The set the model should be evaluated on
        :param y_train: Training set tagged vald be evaluated on
        :param X_test: Test set tagged vald be evaluated on
        :param y_test: Test set tagged values (labels)
        :param data_loader: DataLoader instance used to load data
        :param dataset_version: Version of raw dataset used
        :param evaluator: Logic for model and results evaluation
        :param log_experiment: Whether to log this experiment
        into the experimentation service or not
        :param experiment_logging: Experimentation service instance
        (e.g. MlflowExperimentation)
        :param experiment_name: Name of experiment,
        to be used by the experimentation service

        :example:

        # Call experiment runner and log all objects' parameters:
        experiment_runner = ExperimentRunner(
            model=mock_model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            data_loader=data_loader,
            evaluator=evaluator,
            experiment_logger=experiment_logger,
            experiment_name="Text",
            )

        # Option 1: Fit, predict and evaluate :
        results = experiment_runner.run()
        print(results)

        # Option 2: Predict and evaluate ((model is already fitted):
        results = experiment_runner.evaluate()
        print(results)

        # Option 3: Run each part separately:
        experiment_runner.fit_model()
        experiment_runner.predict()
        results = experiment_runner.evaluate()

        """
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.data_loader = data_loader
        self.evaluator = evaluator
        self.experiment_logger = experiment_logger
        self.log_experiment = log_experiment
        self.experiment_name = experiment_name
        self._evaluation_metrics = []  # Metrics gathered during experiment
        self._predictions = []  # Predictions gathered during experiment

        logger.info(f"Starting experiment: {self.experiment_name}...")

        if self.log_experiment:
            if not self.experiment_logger:
                raise ValueError(
                    "Experimentation system not passed, cannot log experiment"
                )

            if not self.experiment_name:
                raise ValueError(
                    "Experiment name must be specified for the experiment logging system"
                )

            self.additional_params = experiment_params_to_log

            self.log(data_loader, evaluator, model)

    def log(self, data_loader, evaluator, model):
        """
        Stores the parameters and metrics from the various modules
        and those that were added to ExperimentRunner as kwargs,
        into the experiment logger module.
        """
        logger.info(f"Connecting to {self.experiment_logger.name}")
        self.experiment_logger.set_experiment(name=self.experiment_name)
        self.experiment_logger.start_run()
        self._log_loggable_object(model, "Model")
        self._log_loggable_object(evaluator, "Evaluator")
        self._log_loggable_object(data_loader, "DataLoader")
        if model.preprocessor:
            self._log_loggable_object(model.preprocessor, "Preprocessor")
        if model.postprocessor:
            self._log_loggable_object(model.postprocessor, "Postprocessor")
        # Log additional inputs to this class
        if self.additional_params:
            logger.info(
                f"Logging these additional parameters as well: {self.additional_params}"
            )
            self.experiment_logger.log_params(self.additional_params)

    def _log_loggable_object(self, loggable_object: LoggableObject, object_name):
        if loggable_object:
            params = loggable_object.get_params()
            metrics = loggable_object.get_metrics()

            self.experiment_logger.log_params(params if params else {})
            self.experiment_logger.log_metrics(metrics if metrics else {})
            self.experiment_logger.log_param(object_name, loggable_object.name)

    def run(self) -> EvaluationMetrics:
        """
        Performs model fitting and evaluation
        :return: evaluation results
        """
        self.fit_model()

        self.predict()

        evaluation_result = self.evaluate()

        return evaluation_result

    def fit_model(self) -> None:
        logger.info(f"Fitting model {self.model.name} on {len(self.X_train)} samples")

        self.model.fit(X=self.X_train, y=self.y_train)

    def predict(self):
        """
        Calls the model predict function with the input X_test
        :return: None
        """
        logger.info(
            f"Running model.predict() using model {self.model.name} "
            f"on {len(self.X_test)} test samples"
        )
        self._predictions = self.model.predict(X=self.X_test)

    def evaluate(self) -> EvaluationMetrics:
        """
        Runs evaluation on the given model and test set
        :return: EvaluationResult
        """

        if self._predictions is None or len(self._predictions) == 0:
            logger.info("Predictions not found, running model.predict")
            self.predict()
        else:
            logger.info("Predictions found, skipping model.predict call")

        evaluation_result = self.evaluator.evaluate(self.y_test, self._predictions)
        if self.log_experiment:
            if isinstance(evaluation_result, StepEvaluationMetrics):
                for step in evaluation_result.get_steps():
                    step_metrics = evaluation_result.get_metrics(step=step)
                    self.experiment_logger.log_metrics(step_metrics)
            else:
                self.experiment_logger.log_evaluation_result(evaluation_result)

        self._evaluation_metrics = evaluation_result
        return self._evaluation_metrics

    def get_predictions(self):
        """
        Get already calculated predictions.
        :return: Model predictions on X_test
        """
        if self._predictions is None:
            logger.info(
                "Model was not predicted. "
                "Make sure you called `predict_model()` to calculate predictions"
            )
            return None
        else:
            return self._predictions

    def get_evaluation_metrics(self):
        if self._evaluation_metrics is None:
            logger.info(
                "Evaluation metrics are empty. "
                "Make sure you ran a full experiment (train, predict and evaluate) "
                "prior to calling this method."
            )
            return None
        else:
            return self._evaluation_metrics
