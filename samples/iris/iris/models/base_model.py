import logging
import pickle
from abc import abstractmethod

from iris import LoggableObject
from iris.data_processing import DataProcessor
from iris.experimentation import Experimentation


class BaseModel(LoggableObject):
    """
    Abstract class for a model with unified interface
    """

    def __init__(
        self,
        model_name=None,
        experiment_logger: Experimentation = None,
        preprocessor: DataProcessor = None,
        postprocessor: DataProcessor = None,
        **hyper_params,
    ):
        """
        :param model_name: Model name, to be used by the experiment manager
        :param experiment_logger: Experimentation service for logging model metric during training/inference.
        To be used in the model's functions, e.g. self.experimentation.log_metric("loss", loss)
        :param preprocessor: Preprocessor object that would preprocess each input sample
        :param postprocessor: Postprocessor object that would postprocess data after training/inference
        :param hyper_params: any specific parameter for the model should pass here to be logged
        into the experiment logger
        """
        if model_name:
            self.name = model_name
        else:
            self.name = self.__class__.__name__

        self.preprocessor = preprocessor
        self.postprocessor = postprocessor

        self.hyper_params = hyper_params
        self.experiment_logger = experiment_logger

        super().__init__(name=self.name)

        logging.info(
            f"Created model {self.name} " f"with hyperparams {self.hyper_params}"
        )

    @abstractmethod
    # pylint: disable=invalid-name
    def fit(self, X, y=None) -> None:
        """
        Trains/fits a model. Parameters to the fit function should be added
        via the constructor to verify that they are logged on the experiment logger.
        :param X Training set
        :param y Target values
        :return: None
        """
        pass

    @abstractmethod
    # pylint: disable=invalid-name
    def predict(self, X):
        """
        Run prediction on a new set. Actual implementation,
        parameters and return value should be defined in sub class
        :param X dataset to run prediction on
        """
        pass

    def get_params(self):
        """
        Return the model hyper parameters for logging purposes
        :return:
        """

        return self.hyper_params

    def get_metrics(self):
        # Models are not likely to contain metric values.
        # Override this method if that's true in your case.
        # For logging training metrics, use self.experiment_logger directly
        pass

    def __repr__(self):
        return f"Model: {self.name}"

    def save(self, file_path: str):
        """
        Stores a model in a pickle. Note that some objects are not pickable.
        In such case the save method should be overridden.
        :param file_path: Path to pickle
        :return:
        """
        with open(file_path, "wb+") as file:
            pickle.dump(self, file=file)

    @classmethod
    def load(cls, file_path):
        """
        Loads a model from pickle. Note that some objects are not pickable.
        In such case the load method should be overridden.
        :param file_path: Path to pickle file
        :return: An model of type BaseModel
        """
        with open(file_path, "rb") as file:
            obj = pickle.load(file)
        return obj
