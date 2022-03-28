import os

import tensorflow as tf
from models.model_parameters import ModelPathConfig
from models.parse_args import create_parse_args_hierarchical_bert
from trainers.model_trainer import AbstractConceptEmbeddingTrainer
from utils.model_utils import tokenize_multiple_fields
from models.hierachical_phenotype_model import create_probabilistic_phenotype_model
from models.custom_layers import get_custom_objects
from data_generators.data_generator_base import *
from data_generators.data_classes import TokenizeFieldInfo

from keras_transformer.bert import (
    masked_perplexity, MaskedPenalizedSparseCategoricalCrossentropy)

from tensorflow.keras import optimizers


class ProbabilisticPhenotypeTrainer(AbstractConceptEmbeddingTrainer):
    confidence_penalty = 0.1

    def __init__(self, tokenizer_path: str, embedding_size: int, depth: int,
                 max_num_visits: int, max_num_concepts: int, num_heads: int,
                 time_embeddings_size: int, *args, **kwargs):

        self._tokenizer_path = tokenizer_path
        self._embedding_size = embedding_size
        self._depth = depth
        self._max_num_visits = max_num_visits
        self._max_num_concepts = max_num_concepts
        self._num_heads = num_heads
        self._time_embeddings_size = time_embeddings_size

        super(ProbabilisticPhenotypeTrainer, self).__init__(*args, **kwargs)

        self.get_logger().info(
            f'{self} will be trained with the following parameters:\n'
            f'tokenizer_path: {tokenizer_path}\n'
            f'embedding_size: {embedding_size}\n'
            f'depth: {depth}\n'
            f'max_num_visits: {max_num_visits}\n'
            f'max_num_concepts: {max_num_concepts}\n'
            f'num_heads: {num_heads}\n'
            f'time_embeddings_size: {time_embeddings_size}')

    def _load_dependencies(self):

        self._training_data['patient_concept_ids'] = self._training_data.concept_ids \
            .apply(lambda visit_concepts: np.hstack(visit_concepts))
        tokenize_fields_info = [TokenizeFieldInfo(column_name='patient_concept_ids'),
                                TokenizeFieldInfo(column_name='time_interval_atts')]
        self._tokenizer = tokenize_multiple_fields(
            self._training_data,
            tokenize_fields_info,
            self._tokenizer_path,
            encode=False)

    def create_data_generator(self) -> ProbabilisticPhenotypeDataGenerator:

        parameters = {
            'training_data': self._training_data,
            'concept_tokenizer': self._tokenizer,
            'batch_size': self._batch_size,
            'max_num_of_visits': self._max_num_visits,
            'max_num_of_concepts': self._max_num_concepts
        }

        data_generator_class = ProbabilisticPhenotypeDataGenerator
        return data_generator_class(**parameters)

    def _create_model(self):
        strategy = tf.distribute.MirroredStrategy()
        self.get_logger().info('Number of devices: {}'.format(
            strategy.num_replicas_in_sync))
        with strategy.scope():
            existing_model_path = os.path.join(self.get_model_folder(), 'bert_model.h5')
            if os.path.exists(existing_model_path):
                self.get_logger().info(
                    f'The {self} model will be loaded from {existing_model_path}')
                model = tf.keras.models.load_model(
                    existing_model_path, custom_objects=get_custom_objects())
            else:
                optimizer = optimizers.Adam(lr=self._learning_rate,
                                            beta_1=0.9,
                                            beta_2=0.999,
                                            epsilon=1e-5,
                                            clipnorm=1.0)

                model = create_probabilistic_phenotype_model(
                    num_of_visits=self._max_num_visits,
                    num_of_concepts=self._max_num_concepts,
                    concept_vocab_size=self._tokenizer.get_vocab_size(),
                    embedding_size=self._embedding_size,
                    depth=self._depth,
                    num_heads=self._num_heads,
                    time_embeddings_size=self._time_embeddings_size
                )

                losses = {
                    'concept_predictions':
                        MaskedPenalizedSparseCategoricalCrossentropy(
                            self.confidence_penalty),
                    'condition_predictions':
                        MaskedPenalizedSparseCategoricalCrossentropy(
                            self.confidence_penalty)
                }
                model.compile(
                    optimizer,
                    loss=losses,
                    metrics={'concept_predictions': masked_perplexity})
        return model

    def eval_model(self):
        pass


def main(args):
    config = ModelPathConfig(args.input_folder, args.output_folder)
    ProbabilisticPhenotypeTrainer(
        training_data_parquet_path=config.parquet_data_path,
        model_path=config.model_path,
        tokenizer_path=config.tokenizer_path,
        embedding_size=args.embedding_size,
        depth=args.depth,
        max_num_visits=args.max_num_visits,
        max_num_concepts=args.max_num_concepts,
        num_heads=args.num_heads,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        time_embeddings_size=args.time_embeddings_size,
        use_dask=args.use_dask,
        tf_board_log_path=args.tf_board_log_path).train_model()


if __name__ == "__main__":
    main(create_parse_args_hierarchical_bert().parse_args())
