import copy
from pathlib import Path
from typing import Tuple

import requests
from flair.datasets import CONLL_03

from ner_sample.data import DataLoader


class ConllDataLoader(DataLoader):
    def __init__(
        self,
        dataset_name="conll_03",
        dataset_version="1",
        local_data_path="../data/processed/",
        dataset_path="https://raw.githubusercontent.com/glample/tagger/master/dataset/",
        downsample=0.05
    ):
        """
        Data Loader for the CONLL 03 dataset.
        download_dataset downloads the three datasets (train, testa and testb) from Github
        get_dataset returns a flair Corpus object holding the three datasets.
        """
        self.folds = ("eng.train", "eng.testa", "eng.testb")
        self.local_data_path = local_data_path
        self.dataset_path = dataset_path
        self.downsample = downsample
        super().__init__(dataset_name=dataset_name, dataset_version=dataset_version, downsample=downsample)

    def download_dataset(self) -> None:
        if self.dataset_name == "conll_03" and self.dataset_version == "1":

            for fold in self.folds:
                local_path = Path(self.local_data_path, self.dataset_name).resolve()
                fold_path = self.dataset_path + fold
                if not local_path.exists():
                    local_path.mkdir(parents=True)

                dataset_file = Path(local_path, fold)
                if dataset_file.exists():
                    print("Dataset already exists, skipping download")
                    return

                response = requests.get(fold_path)
                dataset_raw = response.text
                with open(dataset_file, "w") as f:
                    f.write(dataset_raw)
                print(f"Finished writing fold {fold} to {self.local_data_path}")

            print(
                f"Finished downloading dataset {self.dataset_name} version {self.dataset_version}"
            )

        else:
            raise ValueError("Selected dataset was not found")

    def get_dataset(self) -> Tuple:
        try:
            corpus = CONLL_03(base_path=self.local_data_path, in_memory=True)
            corpus = corpus.downsample(self.downsample)  # Just for example purposes

            train = corpus  # includes train and dev

            test = corpus.test

            # Copy labels to a new tag (Flair overrides the ner tag during prediction)
            for sentence in test:
                for token in sentence.tokens:
                    token.annotation_layers["gold_ner"] = copy.deepcopy(
                        token.annotation_layers["ner"]
                    )
                    token.annotation_layers["ner"][0].value = "O"

            return train, test

        except FileNotFoundError:
            print(
                f"Dataset {self.dataset_name} with version {self.dataset_version} not found in data/raw"
            )
