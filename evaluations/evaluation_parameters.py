FULL = 'full'
SEQUENCE_MODEL = 'sequence_model'
BASELINE_MODEL = 'baseline_model'
EVALUATION_CHOICES = [FULL, SEQUENCE_MODEL, BASELINE_MODEL]
LSTM = 'lstm'
VANILLA_BERT_LSTM = 'vanilla_bert_lstm'
PROBABILISTIC_BERT_LSTM = 'probabilistic_bert_lstm'
PROBABILISTIC_PHENOTYPE_LSTM = 'probabilistic_phenotype_lstm'
VANILLA_BERT_FEED_FORWARD = 'vanilla_bert_feed_forward'
SLIDING_BERT = 'sliding_bert'
TEMPORAL_BERT_LSTM = 'temporal_bert_lstm'
RANDOM_VANILLA_BERT_LSTM = 'random_vanilla_bert_lstm'
HIERARCHICAL_BERT_LSTM = 'hierarchical_bert_lstm'
HIERARCHICAL_BERT_POOLING = 'hierarchical_bert_pooling'
RANDOM_HIERARCHICAL_BERT_LSTM = 'random_hierarchical_bert_lstm'
SEQUENCE_MODEL_EVALUATORS = [LSTM, VANILLA_BERT_LSTM, VANILLA_BERT_FEED_FORWARD, TEMPORAL_BERT_LSTM,
                             SLIDING_BERT, RANDOM_VANILLA_BERT_LSTM, HIERARCHICAL_BERT_LSTM,
                             RANDOM_HIERARCHICAL_BERT_LSTM, PROBABILISTIC_BERT_LSTM,
                             PROBABILISTIC_PHENOTYPE_LSTM, HIERARCHICAL_BERT_POOLING]
