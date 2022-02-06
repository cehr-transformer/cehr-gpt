from models.custom_layers import *


def transformer_hierarchical_bert_model(num_of_visits,
                                        num_of_concepts,
                                        concept_vocab_size,
                                        embedding_size,
                                        depth: int,
                                        num_heads: int,
                                        num_of_exchanges: int,
                                        transformer_dropout: float = 0.1,
                                        embedding_dropout: float = 0.6,
                                        l2_reg_penalty: float = 1e-4,
                                        time_embeddings_size: int = 16,
                                        include_second_tiered_learning_objectives: bool = False,
                                        visit_vocab_size: int = None):
    """
    Create a hierarchical bert model

    :param num_of_visits:
    :param num_of_concepts:
    :param concept_vocab_size:
    :param embedding_size:
    :param depth:
    :param num_heads:
    :param num_of_exchanges:
    :param transformer_dropout:
    :param embedding_dropout:
    :param l2_reg_penalty:
    :param time_embeddings_size:
    :param include_second_tiered_learning_objectives:
    :param visit_vocab_size:
    :return:
    """
    # If the second tiered learning objectives are enabled, visit_vocab_size needs to be provided
    if include_second_tiered_learning_objectives and not visit_vocab_size:
        raise RuntimeError(f'visit_vocab_size can not be null '
                           f'when the second learning objectives are enabled')

    pat_seq = tf.keras.layers.Input(
        shape=(num_of_visits, num_of_concepts,),
        dtype='int32',
        name='pat_seq'
    )
    pat_seq_age = tf.keras.layers.Input(
        shape=(num_of_visits, num_of_concepts,),
        dtype='int32',
        name='pat_seq_age'
    )
    pat_seq_time = tf.keras.layers.Input(
        shape=(num_of_visits, num_of_concepts,),
        dtype='int32',
        name='pat_seq_time'
    )
    pat_mask = tf.keras.layers.Input(
        shape=(num_of_visits, num_of_concepts,),
        dtype='int32',
        name='pat_mask'
    )
    visit_segment = tf.keras.layers.Input(
        shape=(num_of_visits,),
        dtype='int32',
        name='visit_segment'
    )

    visit_mask = tf.keras.layers.Input(
        shape=(num_of_visits,),
        dtype='int32',
        name='visit_mask')

    visit_time_delta_att = tf.keras.layers.Input(
        shape=(num_of_visits - 1,),
        dtype='int32',
        name='visit_time_delta_att'
    )

    visit_rank_order = tf.keras.layers.Input(
        shape=(num_of_visits,),
        dtype='int32',
        name='visit_rank_order')

    # Create a list of inputs so the model could reference these later
    default_inputs = [pat_seq, pat_seq_age, pat_seq_time, pat_mask,
                      visit_segment, visit_mask, visit_time_delta_att,
                      visit_rank_order]

    # Expand dimensions for masking MultiHeadAttention in Concept Encoder
    pat_concept_mask = tf.reshape(
        pat_mask,
        shape=(-1, num_of_concepts)
    )[:, tf.newaxis, tf.newaxis, :]

    # Expand dimension for masking MultiHeadAttention in Visit Encoder
    visit_mask_with_att = (tf.reshape(
        tf.stack([visit_mask, visit_mask], axis=2),
        shape=(-1, num_of_visits * 2)
    )[:, 1:])[:, tf.newaxis, tf.newaxis, :]

    # output the embedding_matrix:
    l2_regularizer = (tf.keras.regularizers.l2(l2_reg_penalty) if l2_reg_penalty else None)
    concept_embedding_layer = ReusableEmbedding(
        concept_vocab_size,
        embedding_size,
        name='bpe_embeddings',
        embeddings_regularizer=l2_regularizer
    )

    # define the time embedding layer for absolute time stamps (since 1970)
    time_embedding_layer = TimeEmbeddingLayer(
        embedding_size=time_embeddings_size,
        name='time_embedding_layer'
    )
    # define the age embedding layer for the age w.r.t the medical record
    age_embedding_layer = TimeEmbeddingLayer(
        embedding_size=time_embeddings_size,
        name='age_embedding_layer'
    )
    # define positional encoding layer for visit numbers, the visit numbers are normalized
    # by subtracting visit numbers off the first visit number
    positional_encoding_layer = PositionalEncodingLayer(
        max_sequence_length=num_of_visits * num_of_concepts,
        embedding_size=time_embeddings_size,
        name='positional_encoding_layer'
    )
    # Temporal transformation
    temporal_transformation_layer = tf.keras.layers.Dense(
        embedding_size,
        activation='tanh',
        name='temporal_transformation'
    )

    # Look up the embeddings for the concepts
    concept_embeddings, embedding_matrix = concept_embedding_layer(
        pat_seq
    )

    # Look up the embeddings for the att tokens
    att_embeddings, _ = concept_embedding_layer(
        visit_time_delta_att
    )

    pt_seq_age_embeddings = age_embedding_layer(
        pat_seq_age
    )
    pt_seq_time_embeddings = time_embedding_layer(
        pat_seq_time
    )
    visit_positional_encoding = positional_encoding_layer(
        visit_rank_order
    )
    visit_positional_encoding = tf.tile(
        visit_positional_encoding[:, :, tf.newaxis, :], [1, 1, num_of_concepts, 1])

    # (batch, num_of_visits, num_of_concepts, embedding_size)
    concept_embeddings = temporal_transformation_layer(
        tf.concat(
            [concept_embeddings,
             pt_seq_age_embeddings,
             pt_seq_time_embeddings,
             visit_positional_encoding],
            axis=-1, name='concat_for_encoder'
        )
    )

    # The first bert applied at the visit level
    concept_encoder = Encoder(
        name='concept_encoder',
        num_layers=depth,
        d_model=embedding_size,
        num_heads=num_heads,
        dropout_rate=transformer_dropout
    )

    concept_embeddings = tf.reshape(
        concept_embeddings,
        shape=(-1, num_of_concepts, embedding_size)
    )

    concept_embeddings, _ = concept_encoder(
        concept_embeddings,  # be reused
        pat_concept_mask  # not change
    )

    # (batch_size, num_of_visits, num_of_concepts, embedding_size)
    concept_embeddings = tf.reshape(
        concept_embeddings,
        shape=(-1, num_of_visits, num_of_concepts, embedding_size)
    )

    # Step 2 generate visit embeddings
    # Slice out the first contextualized embedding of each visit
    # (batch_size, num_of_visits, embedding_size)
    visit_embeddings = concept_embeddings[:, :, 0]

    # Insert the att embeddings between the visit embeddings using the following trick
    identity = tf.constant(
        np.insert(
            np.identity(num_of_visits),
            obj=range(1, num_of_visits),
            values=0,
            axis=1
        ),
        dtype=tf.float32
    )

    # (batch_size, num_of_visits + num_of_visits - 1, embedding_size)
    expanded_visit_embeddings = tf.transpose(
        tf.transpose(visit_embeddings, perm=[0, 2, 1]) @ identity,
        perm=[0, 2, 1]
    )

    # Create the inverse "identity" matrix for inserting att embeddings
    identity_inverse = tf.constant(
        np.insert(
            np.identity(num_of_visits - 1),
            obj=range(0, num_of_visits),
            values=0,
            axis=1),
        dtype=tf.float32)

    # (batch_size, num_of_visits + num_of_visits - 1, embedding_size)
    expanded_att_embeddings = tf.transpose(
        tf.transpose(att_embeddings, perm=[0, 2, 1]) @ identity_inverse,
        perm=[0, 2, 1]
    )

    # Insert the att embeddings between visit embeddings
    # (batch_size, num_of_visits + num_of_visits - 1, embedding_size)
    contextualized_visit_embeddings = expanded_visit_embeddings + expanded_att_embeddings

    # Second bert applied at the patient level to the visit embeddings
    visit_encoder = Encoder(name='visit_encoder',
                            num_layers=depth,
                            d_model=embedding_size,
                            num_heads=num_heads,
                            dropout_rate=transformer_dropout)

    # Feed augmented visit embeddings into encoders to get contextualized visit embeddings
    contextualized_visit_embeddings, _ = visit_encoder(
        contextualized_visit_embeddings,
        visit_mask_with_att
    )

    # # Step 3 decoder applied to patient level
    # Reshape the data in visit view back to patient view:
    # (batch, num_of_visits * num_of_concepts, embedding_size)
    concept_embeddings = tf.reshape(
        concept_embeddings,
        shape=(-1, num_of_visits * num_of_concepts, embedding_size)
    )

    # Let local concept embeddings access the global representatives of each visit
    multi_head_attention_layer = MultiHeadAttention(
        d_model=embedding_size,
        num_heads=num_heads
    )

    layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    dropout1 = tf.keras.layers.Dropout(transformer_dropout)

    global_concept_embeddings, _ = multi_head_attention_layer(
        contextualized_visit_embeddings,
        contextualized_visit_embeddings,
        concept_embeddings,
        visit_mask_with_att)

    global_concept_embeddings = layernorm1(dropout1(global_concept_embeddings))

    ffn = point_wise_feed_forward_network(embedding_size, 512)
    layernorm2 = tf.keras.layers.LayerNormalization(
        epsilon=1e-6,
        name='global_concept_embeddings_normalization'
    )
    dropout2 = tf.keras.layers.Dropout(transformer_dropout)
    global_concept_embeddings = layernorm2(dropout2(ffn(global_concept_embeddings)))

    concept_output_layer = TiedOutputEmbedding(
        projection_regularizer=l2_regularizer,
        projection_dropout=embedding_dropout,
        name='concept_prediction_logits')

    concept_softmax_layer = tf.keras.layers.Softmax(
        name='concept_predictions'
    )

    concept_predictions = concept_softmax_layer(
        concept_output_layer([global_concept_embeddings, embedding_matrix])
    )

    outputs = [concept_predictions]

    if include_second_tiered_learning_objectives:
        # Slice out the the visit embeddings (CLS tokens)
        visit_embeddings_without_att = identity @ contextualized_visit_embeddings

        visit_prediction_dense = tf.keras.layers.Dense(
            visit_vocab_size,
            name='visit_prediction_dense'
        )

        visit_softmax_layer = tf.keras.layers.Softmax(
            name='visit_predictions'
        )

        visit_predictions = visit_softmax_layer(
            visit_prediction_dense(visit_embeddings_without_att)
        )

        outputs.extend([visit_predictions])

    hierarchical_bert = tf.keras.Model(
        inputs=default_inputs,
        outputs=outputs)

    return hierarchical_bert