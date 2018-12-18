import tensorflow as tf

tf.app.flags.DEFINE_string('chatbot_data_path', './data/ChatbotData.csv', 'chatbot data path')
tf.app.flags.DEFINE_string('embedding_matrix_path', './data/embedding.npy', 'embedding matrix path')
tf.app.flags.DEFINE_integer('embedding_dim', 300, 'embedding dim')
tf.app.flags.DEFINE_float('dropout_rate', 0.1, 'dropout rate')
tf.app.flags.DEFINE_string('check_point', './check_point', 'chech_point')
tf.app.flags.DEFINE_integer('batch_size', 16, 'batch size')
tf.app.flags.DEFINE_integer('epoch', 1, 'epoch')
tf.app.flags.DEFINE_integer('num_units', 300, 'num units')
tf.app.flags.DEFINE_float('learning_rate', 0.005, 'learning rate')


DEFINES = tf.app.flags.FLAGS
