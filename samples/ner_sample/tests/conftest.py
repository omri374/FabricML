import os
from pathlib import Path

import pytest
from flair.models import SequenceTagger

from ner_sample.models.flair_ner import FlairNERModel
from tests.mocks import MockDataLoader


@pytest.fixture
def dataset_loader():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    mock_data_path = Path(dir_path, "resources").resolve()
    return MockDataLoader(local_data_path=mock_data_path)


@pytest.fixture
def pretrained_model(dataset_loader):
    corpus, _ = dataset_loader.get_dataset()

    # Load pretrained model
    model = FlairNERModel(corpus=corpus, max_epochs=1)
    model.tagger = SequenceTagger.load("ner")
    return model
