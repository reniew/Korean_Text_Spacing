import tensorflow as tf

from tensorflow.python.keras.layers import GRU, Bidirectional, Dense, Dropout

from configs import DEFINES

def bidirectional_lstm(inputs):

    with tf.variable_scope('bidirection_gru', reuse = tf.AUTO_REUSE):
        gru = GRU(units = DEFINES.num_units, return_sequences=True)
        gru2 = GRU(units = DEFINES.num_units*2, return_sequences=True)
        bidirectional = Bidirectional(gru)
        bidirectional2 = Bidirectional(gru2)

        output = bidirectional(inputs)
        output = bidirectional2(output)

    return output


def model_fn(mode, features, labels, params):

    TRAIN = mode == tf.estimator.ModeKeys.TRAIN
    EVAL = mode == tf.estimator.ModeKeys.EVAL
    PREDICT = mode == tf.estimator.ModeKeys.PREDICT

    inputs = features['inputs']
    embedding_matrix = params['embedding_matrix']
    dropout = Dropout(DEFINES.dropout_rate)
    dense = Dense(units = params['vocab_size'], activation = tf.nn.sigmoid)

    embed_inputs = tf.nn.embedding_lookup(ids = inputs, params = embedding_matrix)
    embed_inputs = dropout(embed_inputs)


    outputs = bidirectional_lstm(embed_inputs)

    with tf.variable_scope('linear', reuse = tf.AUTO_REUSE):
        logits = dense(outputs)

    predict = tf.cast(tf.round(logits), dtype = tf.int32)


    if PREDICT:
        predictions = {'prediction': predict}
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    y_onehot = tf.one_hot(indices = labels, depth = params['vocab_size'])
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = y_onehot)
    loss = tf.reduce_mean(loss)

    if EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss)

    optimizer = tf.train.AdamOptimizer(learning_rate = DEFINES.learning_rate)
    train_op = optimizer.minimize(loss, global_step = tf.train.get_global_step())


    return tf.estimator.EstimatorSpec(mode, loss = loss, train_op = train_op)

def make_loss(logits, inputs, labels):

    with tf.variable_scope('loss'):
        target_lengths = tf.reduce_sum(
                tf.to_int32(tf.not_equal(inputs, 0)), 1)

        weight_masks = tf.sequence_mask(
                    lengths=target_lengths,
                    maxlen=labels.shape.as_list()[1],
                    dtype=tf.float32, name='masks')

        loss = tf.contrib.seq2seq.sequence_loss(
                logits=logits,
                targets=labels,
                weights=weight_masks,
                name="squence_loss")
    return loss
