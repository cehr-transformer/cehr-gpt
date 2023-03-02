import configparser
import config.parameters as p
from config.grid_search_config import LEARNING_RATE, LSTM_DIRECTION, LSTM_UNIT
from evaluations.evaluation_parameters import *
from evaluations.evaluation_parse_args import create_evaluation_args
from evaluations.model_evaluators.hierarchical_bert_evaluators import *
from evaluations.model_evaluators.bert_model_evaluators import *
from evaluations.model_evaluators.sequence_model_evaluators import *
from evaluations.model_evaluators.frequency_model_evaluators import *
from utils.model_utils import *


def get_grid_search_config(grid_search_config) -> GridSearchConfig:
    """
    Read the grid search config file and load learning_rates, lstm_directions and lstm_units

    :param grid_search_config:
    :return:
    """
    try:
        if grid_search_config:
            config = configparser.ConfigParser()
            config.read(grid_search_config)

            learning_rates = [float(v) for v in dict(config[LEARNING_RATE].items()).values()]
            lstm_directions = [bool(int(v)) for v in dict(config[LSTM_DIRECTION].items()).values()]
            lstm_units = [int(v) for v in dict(config[LSTM_UNIT].items()).values()]

            return GridSearchConfig(
                learning_rates=learning_rates,
                lstm_directions=lstm_directions,
                lstm_units=lstm_units
            )

    except Exception as e:
        print(f'{grid_search_config} cannot be parsed. Error message" {e}')
    else:
        print(f'grid_search_config is not provided, will use the default GridSearchConfig')

    return GridSearchConfig()


def evaluate_sequence_models(args):
    # Load the training data
    dataset = pd.read_parquet(args.sequence_model_data_path)
    logging.getLogger(__name__).info(
        f'sequence_model_data_path: {args.sequence_model_data_path}\n'
        f'args.grid_search_config: {args.grid_search_config}\n'
    )
    grid_search_config = get_grid_search_config(
        args.grid_search_config
    )
    if LSTM in args.model_evaluators:
        validate_folder(args.time_attention_model_folder)
        time_attention_tokenizer_path = os.path.join(
            args.time_attention_model_folder,
            p.tokenizer_path
        )
        time_aware_model_path = os.path.join(
            args.time_attention_model_folder,
            p.time_attention_model_path
        )
        BiLstmModelEvaluator(
            dataset=dataset,
            evaluation_folder=args.evaluation_folder,
            num_of_folds=args.num_of_folds,
            is_transfer_learning=args.is_transfer_learning,
            training_percentage=args.training_percentage,
            max_seq_length=args.max_seq_length,
            embedding_size=args.embedding_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            time_aware_model_path=time_aware_model_path,
            tokenizer_path=time_attention_tokenizer_path,
            sequence_model_name=args.sequence_model_name,
            learning_rate=args.learning_rate,
            cross_validation_test=args.cross_validation_test,
            grid_search_config=grid_search_config,
            is_chronological_test=args.is_chronological_test,
            k_fold_test=args.k_fold_test,
            multiple_test_run=args.multiple_test_run
        ).eval_model()

    if VANILLA_BERT_FEED_FORWARD in args.model_evaluators:
        validate_folder(args.vanilla_bert_model_folder)
        bert_tokenizer_path = os.path.join(args.vanilla_bert_model_folder,
                                           p.tokenizer_path)
        bert_model_path = os.path.join(args.vanilla_bert_model_folder,
                                       p.bert_model_validation_path)
        BertFeedForwardModelEvaluator(
            dataset=dataset,
            evaluation_folder=args.evaluation_folder,
            num_of_folds=args.num_of_folds,
            is_transfer_learning=args.is_transfer_learning,
            training_percentage=args.training_percentage,
            max_seq_length=args.max_seq_length,
            batch_size=args.batch_size,
            epochs=args.epochs,
            bert_model_path=bert_model_path,
            tokenizer_path=bert_tokenizer_path,
            is_temporal=False,
            sequence_model_name=args.sequence_model_name,
            learning_rate=args.learning_rate,
            cross_validation_test=args.cross_validation_test,
            grid_search_config=grid_search_config,
            k_fold_test=args.k_fold_test,
            multiple_test_run=args.multiple_test_run
        ).eval_model()

    if SLIDING_BERT in args.model_evaluators:
        validate_folder(args.vanilla_bert_model_folder)
        bert_tokenizer_path = os.path.join(args.vanilla_bert_model_folder,
                                           p.tokenizer_path)
        bert_model_path = os.path.join(args.vanilla_bert_model_folder,
                                       p.bert_model_validation_path)
        SlidingBertModelEvaluator(
            dataset=dataset,
            evaluation_folder=args.evaluation_folder,
            num_of_folds=args.num_of_folds,
            is_transfer_learning=args.is_transfer_learning,
            training_percentage=args.training_percentage,
            max_seq_length=args.max_seq_length,
            batch_size=args.batch_size,
            epochs=args.epochs,
            bert_model_path=bert_model_path,
            tokenizer_path=bert_tokenizer_path,
            stride=args.stride,
            context_window=args.context_window,
            sequence_model_name=args.sequence_model_name,
            learning_rate=args.learning_rate,
            cross_validation_test=args.cross_validation_test,
            grid_search_config=grid_search_config,
            k_fold_test=args.k_fold_test,
            multiple_test_run=args.multiple_test_run
        ).eval_model()

    if VANILLA_BERT_LSTM in args.model_evaluators:
        validate_folder(args.vanilla_bert_model_folder)
        bert_tokenizer_path = os.path.join(args.vanilla_bert_model_folder,
                                           p.tokenizer_path)
        bert_model_path = os.path.join(args.vanilla_bert_model_folder,
                                       p.bert_model_validation_path)
        BertLstmModelEvaluator(
            dataset=dataset,
            evaluation_folder=args.evaluation_folder,
            num_of_folds=args.num_of_folds,
            is_transfer_learning=args.is_transfer_learning,
            training_percentage=args.training_percentage,
            max_seq_length=args.max_seq_length,
            batch_size=args.batch_size,
            epochs=args.epochs,
            bert_model_path=bert_model_path,
            tokenizer_path=bert_tokenizer_path,
            is_temporal=False,
            sequence_model_name=args.sequence_model_name,
            learning_rate=args.learning_rate,
            cross_validation_test=args.cross_validation_test,
            grid_search_config=grid_search_config,
            is_chronological_test=args.is_chronological_test,
            freeze_pretrained_model=args.freeze_pretrained_model,
            k_fold_test=args.k_fold_test,
            multiple_test_run=args.multiple_test_run
        ).eval_model()

    if RANDOM_VANILLA_BERT_LSTM in args.model_evaluators:
        validate_folder(args.vanilla_bert_model_folder)
        bert_model_path = os.path.join(args.vanilla_bert_model_folder,
                                       p.bert_model_validation_path)
        bert_tokenizer_path = os.path.join(args.vanilla_bert_model_folder,
                                           p.tokenizer_path)
        visit_tokenizer_path = os.path.join(args.vanilla_bert_model_folder,
                                            p.visit_tokenizer_path)
        RandomVanillaLstmBertModelEvaluator(
            dataset=dataset,
            evaluation_folder=args.evaluation_folder,
            num_of_folds=args.num_of_folds,
            is_transfer_learning=args.is_transfer_learning,
            training_percentage=args.training_percentage,
            max_seq_length=args.max_seq_length,
            batch_size=args.batch_size,
            epochs=args.epochs,
            bert_model_path=bert_model_path,
            tokenizer_path=bert_tokenizer_path,
            visit_tokenizer_path=visit_tokenizer_path,
            is_temporal=False,
            sequence_model_name=args.sequence_model_name,
            embedding_size=args.embedding_size,
            depth=args.depth,
            num_heads=args.num_heads,
            use_time_embedding=args.use_time_embedding,
            time_embeddings_size=args.time_embeddings_size,
            learning_rate=args.learning_rate,
            cross_validation_test=args.cross_validation_test,
            grid_search_config=grid_search_config,
            is_chronological_test=args.is_chronological_test,
            freeze_pretrained_model=args.freeze_pretrained_model,
            k_fold_test=args.k_fold_test,
            multiple_test_run=args.multiple_test_run
        ).eval_model()

    if HIERARCHICAL_BERT_LSTM in args.model_evaluators:
        validate_folder(args.vanilla_bert_model_folder)
        bert_model_path = os.path.join(args.vanilla_bert_model_folder,
                                       p.bert_model_validation_path)
        bert_tokenizer_path = os.path.join(args.vanilla_bert_model_folder,
                                           p.tokenizer_path)
        bert_visit_tokenizer_path = os.path.join(
            args.vanilla_bert_model_folder,
            p.visit_tokenizer_path
        )
        HierarchicalBertEvaluator(
            dataset=dataset,
            evaluation_folder=args.evaluation_folder,
            num_of_folds=args.num_of_folds,
            is_transfer_learning=args.is_transfer_learning,
            training_percentage=args.training_percentage,
            max_num_of_visits=args.max_num_of_visits,
            max_num_of_concepts=args.max_num_of_concepts,
            batch_size=args.batch_size,
            epochs=args.epochs,
            bert_model_path=bert_model_path,
            tokenizer_path=bert_tokenizer_path,
            visit_tokenizer_path=bert_visit_tokenizer_path,
            sequence_model_name=args.sequence_model_name,
            learning_rate=args.learning_rate,
            cross_validation_test=args.cross_validation_test,
            grid_search_config=grid_search_config,
            include_att_tokens=args.include_att_tokens,
            is_chronological_test=args.is_chronological_test,
            freeze_pretrained_model=args.freeze_pretrained_model,
            k_fold_test=args.k_fold_test,
            multiple_test_run=args.multiple_test_run
        ).eval_model()

    if HIERARCHICAL_BERT_POOLING in args.model_evaluators:
        validate_folder(args.vanilla_bert_model_folder)
        bert_model_path = os.path.join(args.vanilla_bert_model_folder,
                                       p.bert_model_validation_path)
        bert_tokenizer_path = os.path.join(args.vanilla_bert_model_folder,
                                           p.tokenizer_path)
        bert_visit_tokenizer_path = os.path.join(
            args.vanilla_bert_model_folder,
            p.visit_tokenizer_path
        )
        HierarchicalBertPoolingEvaluator(
            dataset=dataset,
            evaluation_folder=args.evaluation_folder,
            num_of_folds=args.num_of_folds,
            is_transfer_learning=args.is_transfer_learning,
            training_percentage=args.training_percentage,
            max_num_of_visits=args.max_num_of_visits,
            max_num_of_concepts=args.max_num_of_concepts,
            batch_size=args.batch_size,
            epochs=args.epochs,
            bert_model_path=bert_model_path,
            tokenizer_path=bert_tokenizer_path,
            visit_tokenizer_path=bert_visit_tokenizer_path,
            sequence_model_name=args.sequence_model_name,
            learning_rate=args.learning_rate,
            cross_validation_test=args.cross_validation_test,
            grid_search_config=grid_search_config,
            include_att_tokens=args.include_att_tokens,
            is_chronological_test=args.is_chronological_test,
            freeze_pretrained_model=args.freeze_pretrained_model,
            k_fold_test=args.k_fold_test,
            multiple_test_run=args.multiple_test_run
        ).eval_model()

    if RANDOM_HIERARCHICAL_BERT_LSTM in args.model_evaluators:
        validate_folder(args.vanilla_bert_model_folder)
        bert_model_path = os.path.join(args.vanilla_bert_model_folder,
                                       p.bert_model_validation_path)
        bert_tokenizer_path = os.path.join(args.vanilla_bert_model_folder,
                                           p.tokenizer_path)
        bert_visit_tokenizer_path = os.path.join(
            args.vanilla_bert_model_folder,
            p.visit_tokenizer_path
        )
        RandomHierarchicalBertEvaluator(
            dataset=dataset,
            evaluation_folder=args.evaluation_folder,
            num_of_folds=args.num_of_folds,
            is_transfer_learning=args.is_transfer_learning,
            training_percentage=args.training_percentage,
            num_of_exchanges=args.num_of_exchanges,
            max_num_of_visits=args.max_num_of_visits,
            max_num_of_concepts=args.max_num_of_concepts,
            depth=args.depth,
            num_heads=args.num_heads,
            use_time_embedding=args.use_time_embedding,
            time_embeddings_size=args.time_embeddings_size,
            embedding_size=args.embedding_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            bert_model_path=bert_model_path,
            tokenizer_path=bert_tokenizer_path,
            visit_tokenizer_path=bert_visit_tokenizer_path,
            sequence_model_name=args.sequence_model_name,
            learning_rate=args.learning_rate,
            cross_validation_test=args.cross_validation_test,
            grid_search_config=grid_search_config,
            include_att_tokens=args.include_att_tokens,
            is_chronological_test=args.is_chronological_test,
            k_fold_test=args.k_fold_test,
            multiple_test_run=args.multiple_test_run
        ).eval_model()


def evaluate_baseline_models(args):
    # Load the training data
    dataset = pd.read_parquet(args.data_path)

    LogisticRegressionModelEvaluator(
        dataset=dataset,
        evaluation_folder=args.evaluation_folder,
        num_of_folds=args.num_of_folds,
        is_transfer_learning=args.is_transfer_learning,
        training_percentage=args.training_percentage
    ).eval_model()

    XGBClassifierEvaluator(
        dataset=dataset,
        evaluation_folder=args.evaluation_folder,
        num_of_folds=args.num_of_folds,
        is_transfer_learning=args.is_transfer_learning,
        training_percentage=args.training_percentage
    ).eval_model()


def main(args):
    tf.keras.utils.set_random_seed(0)

    if args.action == BASELINE_MODEL or args.action == FULL:
        evaluate_baseline_models(args)

    if args.action == SEQUENCE_MODEL or args.action == FULL:
        evaluate_sequence_models(args)


if __name__ == "__main__":
    main(create_evaluation_args().parse_args())
