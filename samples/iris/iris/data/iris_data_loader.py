from iris.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split


class IrisDataLoader(DataLoader):
    def get_dataset(self):
        train = pd.read_csv(
            f"../data/processed/{self.dataset_name}-{self.dataset_version}-train.csv",
            index_col="Id",
        )
        test = pd.read_csv(
            f"../data/processed/{self.dataset_name}-{self.dataset_version}-test.csv",
            index_col="Id",
        )

        X_train = train.drop("Species", axis=1)
        y_train = train["Species"]
        X_test = test.drop("Species", axis=1)
        y_test = test["Species"]

        print(f"Loaded {len(train)} train and {len(test)} test samples")
        return X_train, y_train, X_test, y_test

    def download_dataset(self):
        pass

    def prep_dataset_for_modeling(self):
        """
        Creates a train/test split of the dataset and stores it in data/processed
        """
        print("Creating train/test split")
        iris = pd.read_csv(f"../data/raw/{self.dataset_name}.csv", index_col="Id")
        train, test = train_test_split(iris, test_size=0.3)
        train.to_csv(
            f"../data/processed/{self.dataset_name}-{self.dataset_version}-train.csv"
        )
        test.to_csv(
            f"../data/processed/{self.dataset_name}-{self.dataset_version}-test.csv"
        )
