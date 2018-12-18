import tensorflow as tf

import data_process as data
import model

from configs import DEFINES

def main(self):

    inputs, labels, t2i, i2t, max_len, embedding_matrix = data.load_data()
    vocab_size = len(t2i)
    params = make_params(max_len, vocab_size, embedding_matrix)
    estimator = tf.estimator.Estimator(model_fn = model.model_fn,
                                        model_dir = DEFINES.check_point,
                                        params = params)
    estimator.train(lambda:data.train_input_fn(inputs, labels))



def make_params(max_len, vocab_size, embedding_matrix):

    params = {}
    params['max_len'] = max_len
    params['vocab_size'] = vocab_size
    params['embedding_matrix'] = embedding_matrix

    return params

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
