from typing import List

from flair.data import Corpus
from flair.embeddings import (
    TokenEmbeddings,
    WordEmbeddings,
    PooledFlairEmbeddings,
    StackedEmbeddings,
)
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from tqdm import tqdm

from ner_sample.models import BaseModel


class FlairNERModel(BaseModel):
    def __init__(
        self,
        corpus: Corpus,
        hidden_size: int = 256,
        pooling: str = "min",
        word_embeddings: str = "glove",
        train_with_dev: bool = True,
        max_epochs: int = 10,
    ):
        """
        NER detector using the Flair NLP package.
        Source: https://github.com/flairNLP/flair/blob/master/resources/docs/EXPERIMENTS.md
        All class inputs (except for the corpus) are model hyper parameters.
        They are then directed to the base class and get logged into the experiment logger
        """
        self.tag_type = "ner"
        self.tag_dictionary = None
        self.tagger = None
        self.embeddings = None

        self.hidden_size = hidden_size
        self.pooling = pooling
        self.word_embeddings = word_embeddings
        self.train_with_dev = train_with_dev
        self.max_epochs = max_epochs

        self.set_tagger_definition(corpus)

        hyper_params = self.get_hyper_params(
            hidden_size=hidden_size,
            pooling=pooling,
            word_embeddings=word_embeddings,
            train_with_dev=train_with_dev,
            max_epochs=max_epochs,
        )

        super().__init__(**hyper_params)

    def fit(self, X, y=None) -> None:
        # initialize trainer
        trainer: ModelTrainer = ModelTrainer(self.tagger, X)

        trainer.train(
            "models/taggers/flair-ner",
            train_with_dev=self.train_with_dev,
            max_epochs=self.max_epochs,
        )

    def predict(self, X):
        tagged_sentences = []
        for sentence in tqdm(X):
            self.tagger.predict(sentence)
            tagged_sentences.append(sentence)
        print(f"Tagged {len(tagged_sentences)} sentences")
        return tagged_sentences

    def get_hyper_params(self, **hyper_params):
        basic_params = {
            param_name: param_value
            for (param_name, param_value) in self.tagger.__dict__.items()
            if type(param_value) in (bool, float, int, str)
        }
        hyper_params.update(basic_params)
        return hyper_params

    def set_embeddings_definition(self):
        """
        Sets the embedding layers used by this tagger
        """
        # initialize embeddings
        embedding_types: List[TokenEmbeddings] = [
            # Word embeddings (default = GloVe)
            WordEmbeddings(self.word_embeddings),
            # contextual string embeddings, forward
            PooledFlairEmbeddings("news-forward", pooling=self.pooling),
            # contextual string embeddings, backward
            PooledFlairEmbeddings("news-backward", pooling=self.pooling),
        ]
        self.embeddings: StackedEmbeddings = StackedEmbeddings(
            embeddings=embedding_types
        )

    def set_tagger_definition(self, corpus: Corpus):
        """
        Returns the definition of the Flair SequenceTagger (the full model)
        :param corpus: Used only for setting the tag_dictionary
        """

        if not self.embeddings:
            self.set_embeddings_definition()
        self.tag_dictionary = corpus.make_tag_dictionary(tag_type=self.tag_type)

        tagger: SequenceTagger = SequenceTagger(
            hidden_size=self.hidden_size,
            embeddings=self.embeddings,
            tag_dictionary=self.tag_dictionary,
            tag_type=self.tag_type,
            use_crf=False,
        )
        self.tagger = tagger
