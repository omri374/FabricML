from abc import abstractmethod
from typing import Dict

from ner_sample import LoggableObject


class DataLoader(LoggableObject):
    """
    Abstract class for obtaining data. By using a fixed dataset loading code, dataset name and version,
    it verifies that the experiment is reproducible.
    Helps avoid cases like read_csv("test123-after-changes-23May2051.csv")
    Also, consider using the cookiecutter data-science data folders (external, interim, processed and raw)
    :param dataset_name: Name of dataset for reproducibility
    :param dataset_version: Version of dataset for reproducibility
    """

    def __init__(self, dataset_name, dataset_version, data_loader_name=None, **data_params):
        self.dataset_name = dataset_name
        self.dataset_version = dataset_version
        self.data_params = data_params
        super().__init__(name=data_loader_name)

    @abstractmethod
    def download_dataset(self) -> None:
        """
        Abstract method for downloading a dataset from a repository or a database

        :return: None
        """
        pass

    @abstractmethod
    def get_dataset(self):
        """
        Abstract method for loading a dataset into memory
        :param dataset_name: Name of dataset to load
        :param dataset_version: Version of dataset to load
        :return: The dataset object
        """
        pass

    def get_params(self) -> Dict:
        """
        Reads the dataset loader configuration during experiment logging
        :return: Dictionary with parameters to load into the experiment logging system
        """
        params = {
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
        }

        if self.data_params:
            params.update(self.data_params)

        return params

    def get_metrics(self):
        # Data loaders are not likely to contain metrics
        pass
