import os
from pathlib import Path

import pytest
from flair.data import Corpus, Token
from flair.models import SequenceTagger

from ner_sample.models.flair_ner import FlairNERModel
from tests.mocks import MockDataLoader


def test_flair_inference_ner_mock_data(pretrained_model, dataset_loader):
    _, test = dataset_loader.get_dataset()

    # Predict on test set
    predictions = pretrained_model.predict(test)

    assert len(predictions) == len(test)


def test_flair_inference_correct(
    pretrained_model: FlairNERModel, dataset_loader: MockDataLoader
):
    _, test = dataset_loader.get_dataset()

    # Replace previous tag to populate prediction

    predictions = pretrained_model.predict(test)

    tp_count = 0
    total_count = 0
    for sentence, prediction in zip(test, predictions):
        for i in range(len(sentence.tokens)):
            pred = prediction.tokens[i].annotation_layers["ner"][0].value
            actual = sentence.tokens[i].annotation_layers["gold_ner"][0].value
            if pred == actual:
                tp_count += 1
            total_count += 1

    assert float(tp_count) / total_count > 0.7
