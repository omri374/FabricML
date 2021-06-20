from ner_sample.data import ConllDataLoader


class MockDataLoader(ConllDataLoader):
    """
    Skips the dataset downloading and uses a mock dataset to create a Corpus
    """

    def __init__(self, local_data_path="/tests/resources/"):
        super().__init__(local_data_path=local_data_path, dataset_path=None, downsample=1)

    def download_dataset(self) -> None:
        pass
