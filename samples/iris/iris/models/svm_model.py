from iris.data_processing import EmptyProcessor
from iris.models import BaseModel
from sklearn import svm


class IrisSVMModel(BaseModel):
    """
    sklearn SVM model wrapper
    """

    def __init__(
        self, features, kernel="linear", label="Species", preprocessor=EmptyProcessor()
    ):
        self.features = features
        self.kernel = kernel
        self.model = None

        super().__init__(
            features=features, label=label, kernel=kernel, preprocessor=preprocessor
        )

    def fit(self, X, y=None) -> None:
        train_X = X[self.features]
        train_y = y

        train_X_processed = self.preprocessor.apply_batch(train_X)

        print("Fitting model")
        self.model = svm.SVC(kernel=self.kernel)
        self.model.fit(train_X_processed, train_y)
        print(f"Finished fitting model {self.model}")

    def predict(self, X):
        test_X = X[self.features]
        test_X_processed = self.preprocessor.apply_batch(test_X)

        print(f"Predicting on {len(test_X)} samples")
        predictions = self.model.predict(test_X_processed)
        print(f"Finished prediction")
        return predictions
