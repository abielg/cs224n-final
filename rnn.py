import tensorflow as tf
from model import Model

tf.nn.rnn_cell.BasicLSTMCell

class Config(object):
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.

    CHECK WHICH VALUES WE ACTUALLY NEED AND MODIFY THEM
    """
    n_features = 36
    n_classes = 3
 #   dropout = 0.5
    embed_size = 200 # 200-d word vectors
    hidden_size = 200 # might change this
    batch_size = 2048 # might change this
    n_epochs = 10
    lr = 0.001
    max_sentence_len = 10 # will need to implement some kind of max summary length

class RNN(object):
	def add_placeholders(self):
		self.inputs_placeholder = tf.placeholder(tf.float32, shape=([None, max_sentence_len, 1]), name="x") # none so that that dimension will be batch size
        
        ######### DOUBLE CHECK SIZE OF LABELS PLACEHOLDER ###############
        self.labels_placeholder = tf.placeholder(tf.float32, shape=([None, 1]), name="y")


    def create_feed_dict(self, inputs_batch, labels_batch=None):
    	feed_dict = {
            self.inputs_placeholder: inputs_batch,
        }
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        return feed_dict


	def add_embedding(self):
	   """Adds an embedding layer that maps from input tokens (integers) to vectors and then
        concatenates those vectors:

        TODO:
            - Create an embedding tensor and initialize it with self.pretrained_embeddings.
            - Use the input_placeholder to index into the embeddings tensor, resulting in a
              tensor of shape (None, max_length, n_features, embed_size).
            - Concatenates the embeddings by reshaping the embeddings tensor to shape
              (None, max_length, n_features * embed_size).

        HINTS:
            - You might find tf.nn.embedding_lookup useful.
            - You can use tf.reshape to concatenate the vectors. See
              following link to understand what -1 in a shape means.
              https://www.tensorflow.org/api_docs/python/array_ops/shapes_and_shaping#reshape.

        Returns:
            embeddings: tf.Tensor of shape (None, max_length, n_features*embed_size)
        
        ### YOUR CODE HERE (~4-6 lines)
        embedding_tensor = tf.Variable(self.pretrained_embeddings)
        lookup_tensor = tf.nn.embedding_lookup(embedding_tensor, self.input_placeholder)
        embeddings = tf.reshape(lookup_tensor, [-1, self.config.max_length, self.config.n_features * self.config.embed_size])
        ### END YOUR CODE
        return embeddings
        """

    def add_op(self):
    	fwd_cell = tf.nn.rnn_cell.LSTMCell(1)
    	bckwd_cell = tf.nn.rnn_cell.LSTMCell(1)
    	x = self.inputs_placeholder

    	outputs, output_state_fw, output_state_bw = tf.contrib.rnn.static_bidirectional_rnn(fwd_cell, bckwd_cell, x)
    	return output[-1] #return the last output(which is actually the )






