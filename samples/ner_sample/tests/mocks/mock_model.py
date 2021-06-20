import copy

from ner_sample.models import BaseModel


class MockModel(BaseModel):
    def __init__(self, model_name=None, **hyper_params):
        self.x = None
        super().__init__(model_name=model_name, **hyper_params)

    def fit(self, X, y=None) -> None:
        self.x = X

    def predict(self, X):
        """
        Predict the label with some noise (mock model)
        """
        counter = 0
        for sentence in X:
            for token in sentence.tokens:
                # use original labels:
                token.annotation_layers["ner"] = copy.deepcopy(
                    token.annotation_layers["gold_ner"]
                )
                # revert some to "O":
                if counter % 3 == 0:
                    token.annotation_layers["ner"][0].value = "O"
                counter += 1

        return X
