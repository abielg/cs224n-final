# same imports as rnn.py. Elliott's first attempt at second iteration:
'''
-bidirectional LSTM encoder
-unidirectional LSTM decoder with attention
-greedy search

'''

import tensorflow as tf
import numpy as np
import logging
from datetime import datetime
import os
import time
from util import Progbar, minibatches
import argparse
from tensorflow.python.platform import gfile

from tensorflow.contrib import rnn
from tensorflow.python.platform import gfile


PAD_ID = 0
SOS_ID = 1 # start symbol
UNK_ID = 2 

logger = logging.getLogger("final_project")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

model_path = ""

# copied straight from our rnn.py ... tune this later
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
	embed_size = 200
	encoder_hidden_size = 200
	# decoder_hidden_size = encoder_hidden_size * 2
	decoder_hidden_size = encoder_hidden_size # using same hidden size for simple attention function. will try bilinear attention function later
	batch_size = 50 # batch size was previously 2048
	n_epochs = 10
	lr = 0.001
	max_sentence_len = 20
	vocab_size = 10000

	def __init__(self):
		self.output_path = "results/{:%Y%m%d_%H%M%S}/".format(datetime.now())
		os.makedirs(self.output_path)
		self.model_output = self.output_path + "model.weights"
		self.log_output = self.output_path + "log"
		self.vocabulary = self.init_vocab()
		self.preds_output = self.output_path + "preds"

	def init_vocab(self): # unsure what this does. Abiel probably made it
		vocab_path = 'data/summarization/vocab.dat'
		if gfile.Exists(vocab_path):
			vocab = []
			with gfile.GFile(vocab_path, mode="r") as f:
				vocab.extend(f.readlines())
			vocab = [line.strip('\n') for line in vocab]
			return vocab

	''' # copied this. prob don't need it
	def __init__(self, size, vocab_dim):
		#self.size = size # refers to state size
		#self.vocab_dim = vocab_dim # what it sounds like
	'''

class BallerModel(object):

	def __init__(self, config): # copied from rnn.py
		self.save_predictions = False
		self.config = config
		loaded = np.load('data/summarization/glove.trimmed.50.npz'.format(self.config.embed_size))
		self.embedding_matrix = loaded['glove']
		self.build()

	def __nonzero__(self): # copied from rnn.py
		return self.save_predictions != 0

	def add_placeholders(self):

		# both of the following saved as IDs. will lookup actual embeddings when needed (i.e. training/testing/loss op)
		self.encoder_inputs_placeholder = tf.placeholder(tf.int32, shape=([None, self.config.max_sentence_len]))
		self.labels_placeholder = tf.placeholder(tf.int32, shape=([None, self.config.max_sentence_len]))


	def add_embedding(self):

		# TODO:
		return

	def add_pred_single_batch_train(self):
		x = self.add_embedding(self.encoder_inputs_placeholder) # [batch_sz, max_sentence_len, embed_sz]
		y = self.add_embedding(self.labels_placeholder) # truth embeddings. [batch_size x max_sentence_length x embed_size]
		sequence_lengths = # [batch_size] tensor of the lengths of each truth sentence in the batch
		encoder_outputs, final_enc_state = self.encode(x, sequence_lengths) # TODO: gather sequence lengths
		preds = self.decode_train(y, encoder_outputs, sequence_lengths, final_enc_state)

		return preds # [batch_size, max_sentence_len, vocab_size]

	def add_pred_single_batch_test(self):
		# TODO: fill this in
		return

	def encode(self, inputs, sequence_length):

		#inputs: [batch_size x max_sentence_length x embed_size]
		#sequence_length: [batch_size] vector containing actual length for each sequence

		# encoder_hidden_size is considered the size of the concatenation of the forward and backward cells
		fw_cell = tf.contrib.rnn.LSTMCell(.5 * self.config.encoder_hidden_size, initializer=tf.contrib.layers.xavier_initializer())
		bckwd_cell = tf.contrib.rnn.LSTMCell(.5 * self.config.encoder_hidden_size, initializer=tf.contrib.layers.xavier_initializer())

		outputs, final_state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bckwd_cell, x, sequence_length=sequence_length, dtype=tf.float32)
		concat_outputs = tf.concat(outputs, 2) # dimension: [batch_size x max_sentence_length x encoder_hidden_size]

		return concat_outputs, final_state # concat_outputs -> attention_values, sequence_length -> attention_values_length
		# concat_outputs: [batch_size, max_sentence_len, output_size (encoder_hidden_size)]


	def decode_train(self, input_embeddings, attention_values, sequence_length, decoder_start_state):

		# for each word in ground truth

		preds = [] # currently saved as a list, where each elem represents one timestep. TODO: reshape to tensor, where max_sentence_len is second dimension

		cur_hidden_state = None
		cur_inputs = None
		for i in xrange(self.config.max_sequence_length + 1) # max_sequence_length iterations (+1 for start symbol). starts at 0

			if (i==0): 
				cur_hidden_state = decoder_start_state   # shape: [batch_size, decoder_hidden_size]
				start_symbol_tensor = tf.constant(value=SOS_ID, dtype=int32, shape=[self.config.batch_size])
				cur_inputs = add_embedding(start_symbol_tensor) # TODO: fairly unsure whether or not this will work. will just try
			else:
				# cur_hidden_state initialized previously at end of for loop
				cur_inputs = input_embeddings[:, i, :] # desired shape: [batch_size, embed_size]. unsure whether or not this will work (might return as numpy array?)

			lstmCell = tf.nn.rnn_cell.LSTMCell(num_units=self.config.decoder_hidden_size, initializer=tf.contrib.layers.xavier_initializer())
			intermediate_hidden_state = lstmCell(cur_inputs, cur_hidden_state) # LSTM output has shape: [batch_size, decoder_hidden_size]

			if (i==0):
				# layer_inputs = intermediate_hidden_state
				next_hidden_state = intermediate_hidden_state
			else:
				attn_scores, context_vector = attention_function_simple(intermediate_hidden_state, attention_values, sequence_length)
				
				layer_inputs = tf.concat([intermediate_hidden_state, context_vector], 1) # size: [batch_size, 2 * decoder_hidden_size]
				next_hidden_state = tf.contrib.layers.fully_connected(inputs=layer_inputs, 1), \
					num_outputs=self.config.decoder_hidden_size, activation_fn=tf.nn.tanh, trainable=True)  # should be trainable as is

				W_out = tf.get_variable("W_out", shape=[self.config.decoder_hidden_size, self.config.vocab_size], initializer=tf.contrib.layers.xavier_initializer())
				b_out = tf.get_variable("b_out", shape=[self.config.vocab_size], initializer=tf.constant_initializer(0.0))

				cur_pred = tf.matmul(cur_hidden_state, W_out) + b_out
				preds += cur_pred

				cur_hidden_state = next_hidden_state

		preds_tensor_form = tf.stack(preds, axis=1) # axis = 1 so that max_sentence_len will be second dimension

		return preds_tensor_form

	def attention_function_simple(cell_output, attention_values, sequence_length): # h{enc_i} dot h{s} ... h{s} represents previous hidden state of decoder

		# cell_output: [batch_size, decoder_hidden_size] (same dimension as concat_outputs) # h(t-1) of decoder
		# attention_values: [batch_size, max_sentence_len, encoder_hidden_size] ... okay bc we're using global attention

	#	W = tf.get_variable(shape=[self.encoder_hidden_size, self.decoder_hidden_size], initializer=tf.contrib.layers.xavier_initializer()) # manually set scope?

	#	first_dot = tf.matmul(attention_values, W) # [batch_size, max_sentence_len, 2 * encoder_hidden_size]
	#	print("shape of first dot: ")
	#	print(first_dot.get_shape)

		# global attention. using bilinear formula: ht dot W dot cell_output
		raw_scores = tf.reduce_sum(attention_values * tf.expand_dims(cell_output), [2]) # [batch_sz, max_sentence_len]
		scores_mask = tf.sequence_mask(lenghts=tf.to_int32(sequence_length), max_len=self.config.max_sentence_len, dtype=tf.float32)
		attn_scores = raw_scores * scores_mask + ((1.0 - scores_mask) * tf.float32.min) # unsure what exactly this does ... seems to be a typecast
		prob_distribution = tf.nn.softmax(attn_scores) # [batch_sz, max_sentence_len]
		context_vector = tf.expand_dims(prob_distribution, 2) * attention_values # multiplies each hidden encoder vector by its attention score
		# context_vectors shape: [batch_sz, max_sentence_len, dec_hidden_size]
		context_vector = tf.reduce_sum(context_vector, 1) # reduced over max_sentence_len dimension bc we're using global attention
		# unsure if we actually need context.set_shape ... shape should be: [batch_size, dec_hidden_size]

		return (prob_distribution, context_vector)


	def add_loss_op(self):



		return

	def add_training_op(self):

		return

	def train_on_batch(self):

		return

	def compute_dev_loss(self);

		return





if __name__ == '__main__':
	config = Config()

	with tf.Graph().as_default():
		model = BallerModel(config)

		init = tf.global_variables_initializer()

		with tf.Session() as session:
			session.run(init)

			cell_output = None
			attention_values = None
			sequence_length = None
			attention_scores, attention_context = model.attention_function_simple(cell_output, attention_values, sequence_length)



