import tensorflow as tf
from model import Model


class Config(object):
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.

    CHECK WHICH VALUES WE ACTUALLY NEED AND MODIFY THEM
    """
    n_features = 36
    n_classes = 3
    dropout = 0.5
    embed_size = 50
    encoder_hidden_size = 200
    decoder_hidden_size = encoder_hidden_size * 2
    batch_size = 100 #batch size was previously 2048 
    n_epochs = 10
    lr = 0.001
    max_sentence_len = 10
    vocab_size = 1000

class RNN(object):
	def add_placeholders(self):
        # might need to change stuff in following line
		self.encoder_inputs_placeholder = tf.placeholder(tf.float32, shape=([None, self.config.max_sentence_len]), name="x") # none so that that dimension will be batch size

        # None dimension will get filled in with batch_size
        self.labels_placeholder = tf.placeholder(tf.int32, shape=([None, self.config.max_sentence_len]), name="y")

        self.mask_placeholder = # need to implement this shit for cost function


    def create_feed_dict(self, inputs_batch, labels_batch=None):
    	feed_dict = {
            self.inputs_placeholder: inputs_batch,
        }
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        return feed_dict


    # pretty sure we don't need this
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
    """
    We need two different functions for training and testing. At training time, the word vectors representing
    the headline are passed in as inputs to the decoder. At test time, the previous decoder output is passed
    into the next decoder cell's input. Function handles a single batch.
    """
    def add_pred_single_batch_train(self):
    	x = self.encoder_inputs_placeholder # must be 1D list of int32 Tensors of shape [batch_size]
        y = self.labels_placeholder # must be 1D list of int32 Tensors of shape [batch_size]

    	cell = tf.nn.rnn_cell.LSTMCell(encoder_hidden_size, initializer=tf.contrib.layers.xavier_initializer())
    	
    	#docs: https://www.tensorflow.org/api_docs/python/tf/contrib/legacy_seq2seq/embedding_attention_seq2seq

        # TODO: will need to convert x and y from matrices to lists before they can be fed into legacy
    	outputs, state = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(x, y, cell, vocab_size, vocab_size, embed_size)
        """
        outputs: A list of the same length as decoder_inputs of 2D Tensors with shape [batch_size x num_decoder_symbols] 
        containing the generated outputs
        """
        return outputs # list (word by word) of 2D tensors: [batch_size, vocab_size]


    # Handles a single batch, returns the outputs
    def add_pred_single_batch_test(self):
        x = self.encoder_inputs_placeholder # must be 1D list of int32 Tensors of shape [batch_size]
        # don't have premade decoder inputs. will feed previous decoder output into next decoder cell's input

        # need to verify that this is initialized correctly
        cell = tf.nn.rnn_cell.LSTMCell(encoder_hidden_size, initializer=tf.contrib.layers.xavier_initializer())
        outputs, state = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(x, y, cell, vocab_size, vocab_size, embed_size, feed_previous=True)

        return outputs

    # assumes we already have padding implemented.


    def add_loss_op(self, preds):
        # what loss function to use? cross entropy
        # input shape?
        # output shape?
        # how often do we backprop?

        """
        preds: [batch_size x vocab_size]
        self.labels_placeholder: [batch_size x max_sentence_length]

        """

        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(preds, self.labels_placeholder)
        ce = tf.boolean_mask(ce, self.mask_placeholder)
        loss = tf.reduce_mean(ce)

        return loss

"""
    def encoder_decoder_train2(self): # second version used to fix problem: inputs of seq2seq.embedding ... must take in ints
        x = self.encoder_inputs_placeholder
        y = self.labels_placeholder

        cell = tf.nn.rnn_cell.LSTMCell(encoder_hidden_size, initializer=tf.contrib.layers.xavier_initializer())

      # commented out so code will compile  enc_hidden_states, output_states = tf.nn.bidirectional_dynamic_rnn(cell, cell, x, sequence_length=??, dtype=tf.float32, time_major=??)
"""

'''
    def encoder(self):
    	fwd_cell = tf.nn.rnn_cell.LSTMCell(encoder_hidden_size, initializer=tf.contrib.layers.xavier_initializer())
    	bckwd_cell = tf.nn.rnn_cell.LSTMCell(encoder_hidden_size, initializer=tf.contrib.layers.xavier_initializer())
    	x = self.inputs_placeholder
    	outputs, output_states = tf.nn.bidirectional_dynamic_rnn(fwd_cell, bckwd_cell, x)
    	return tf.concat(output_states, 2)

   	def decoder(self, first_state):
   		x = self.inputs_placeholder
   		lstm_cell = tf.nn.rnn_cell.LSTMCell(decoder_hidden_size, initializer=tf.contrib.layers.xavier_initializer())
   		
   		tf.nn.seq2seq.attention_decoder(x, )

   		tf.nn.dynamic_rnn(lstm_cell, x, initial_state=first_state)
'''

