

import tensorflow as tf
import numpy as np
import logging
from datetime import datetime
import os
import time
from util import Progbar

from tensorflow.contrib import rnn

PAD_ID = 0
SOS_ID = 1
UNK_ID = 2 

logger = logging.getLogger("final_project")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

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
	batch_size = 3 # batch size was previously 2048
	n_epochs = 10
	lr = 0.001
	max_sentence_len = 20
	vocab_size = 1000

	def __init__(self):
		self.output_path = "results/{:%Y%m%d_%H%M%S}/".format(datetime.now())
		os.makedirs(self.output_path)
		self.model_output = self.output_path + "model.weights"
		self.log_output = self.output_path + "log"

class RNN(object):
	def __init__(self, config):
		self.config = config
		loaded = np.load('data/summarization/glove.trimmed.50.npz'.format(self.config.embed_size))
		self.embedding_matrix = loaded['glove']
		self.build() # called in init, just like is done in hw. unsure whether or not we can use a class function here

	def add_placeholders(self):
		self.encoder_inputs_placeholder = tf.placeholder(tf.int32, shape=([self.config.max_sentence_len, self.config.batch_size]), name="x")
		self.unstacked_labels_placeholder = tf.placeholder(tf.int32, shape=([self.config.batch_size, self.config.max_sentence_len]), name="y")
		self.stacked_labels_placeholder = tf.placeholder(tf.int32, shape=([self.config.max_sentence_len, None]), name="y_stacked")
		# still not sure that we need the 1's above
		self.mask_placeholder = tf.placeholder(tf.bool, shape=([self.config.batch_size, self.config.max_sentence_len])) # batch_sz x max_sentence_length
		#SWITHCED TYPE OF THE PLACEHOLDERS FROM FLOAT TO INT

	def create_feed_dict(self, inputs_batch, unstacked_labels_batch=None, stacked_labels_batch=None, mask_batch=None):
		feed_dict = {
			self.encoder_inputs_placeholder: inputs_batch,
		}
		if unstacked_labels_batch is not None:
			feed_dict[self.unstacked_labels_placeholder] = unstacked_labels_batch

		if stacked_labels_batch is not None:
			feed_dict[self.stacked_labels_placeholder] = stacked_labels_batch

		if mask_batch is not None: # TODO: create mask_batch using ground truth
			feed_dict[self.mask_placeholder] = mask_batch
		return feed_dict


	def add_embedding(self):
		"""Adds an embedding layer that maps from input tokens (integers) to vectors and then
		concatenates those vectors:

			- Create an embedding tensor and initialize it with self.pretrained_embeddings.
			- Use the input_placeholder to index into the embeddings tensor, resulting in a
			  tensor of shape (None, max_length, n_features, embed_size).
			- Concatenates the embeddings by reshaping the embeddings tensor to shape
			  (None, max_length, n_features * embed_size).

		Returns:
			embeddings: tf.Tensor of shape (None, max_length, embed_size)
		"""
		### YOUR CODE HERE (~4-6 lines)
		embedding_tensor = tf.Variable(self.embedding_matrix)
		lookup_tensor = tf.nn.embedding_lookup(embedding_tensor, self.inputs_placeholder)
		embeddings = tf.reshape(lookup_tensor, [-1, self.config.max_sentence_len, self.config.embed_size])
		print("Created embeddings tensor for the input")
		### END YOUR CODE
		return embeddings
		

	""" ELLIOTTS ADDITIONS
	We need two different functions for training and testing. At training time, the word vectors representing
	the headline are passed in as inputs to the decoder. At test time, the previous decoder output is passed
	into the next decoder cell's input. Function handles a single batch.
	"""

	def add_pred_single_batch_train2(self):
		x = self.encoder_inputs_placeholder # float32 Tensor of shape [batch_size, max_sentence_length, embed_size]
		y = self.stacked_labels_placeholder_list # int32 Tensor of shape [batch_size, max_sentence_length, embed_size]
		#encoder_sequence_length = # TODO: fill this in

		fw_cell = tf.contrib.rnn.LSTMCell(self.config.encoder_hidden_size, initializer=tf.contrib.layers.xavier_initializer())
		bckwd_cell = tf.contrib.rnn.LSTMCell(self.config.encoder_hidden_size, initializer=tf.contrib.layers.xavier_initializer())

		#docs: https://www.tensorflow.org/api_docs/python/tf/contrib/legacy_seq2seq/embedding_attention_seq2seq


		outputs, output_states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bckwd_cell, x, sequence_length=encoder_sequence_length, dtype=tf.float32)
		encoder_final_states = tf.concat(output_states, 2) # dimension: [batch_size x max_sentence_length x encoder_hidden_size * 2]

		decoder_cell = tf.contrib.rnn.LSTMCell(self.config.decoder_hidden_size, initializer=tf.contrib.layers.xavier_initializer())

		#decoder_sequence_length = 
		outputs, state = tf.nn.dynamic_rnn(decoder_cell, y, sequence_length=decoder_sequence_length, initial_state=encoder_final_states)

		W = tf.Variable("W", shape=[None, self.config.decoder_hidden_size, self.config.vocab_size], initializer=tf.contrib.layers.xavier_initializer())
		b = tf.Variable("b", shape=[None, self.config.max_sentence_len, self.config.vocab_size], initializer=tf.constant_initializer(0.0))

		preds = tf.matmul(outputs, W) + b

		"""
		outputs: A a tensor shaped [batch_size, max_sentence_len, decoder_hidden_size]
		containing the generated outputs
		"""
		return preds # list (word by word) of 2D tensors: [batch_size, max_sentence_len, vocab_size]


	def add_pred_single_batch_test2(self):
		x = self.encoder_inputs_placeholder # float32 Tensor of shape [batch_size, max_sentence_length, embed_size]
		y = self.stacked_labels_placeholder_list # int32 Tensor of shape [batch_size, max_sentence_length, embed_size]
		#encoder_sequence_length = # TODO: fill this in

		fw_cell = tf.contrib.rnn.LSTMCell(self.config.encoder_hidden_size, initializer=tf.contrib.layers.xavier_initializer())
		bckwd_cell = tf.contrib.rnn.LSTMCell(self.config.encoder_hidden_size, initializer=tf.contrib.layers.xavier_initializer())

		#docs: https://www.tensorflow.org/api_docs/python/tf/contrib/legacy_seq2seq/embedding_attention_seq2seq

		outputs, output_states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bckwd_cell, x, sequence_length=encoder_sequence_length, dtype=tf.float32)
		encoder_final_states = tf.concat(output_states, 2) # dimension: [batch_size x max_sentence_length x encoder_hidden_size * 2]

		decoder_cell = tf.contrib.rnn.LSTMCell(self.config.decoder_hidden_size, initializer=tf.contrib.layers.xavier_initializer())

		#decoder_sequence_length = 
		outputs, state = tf.nn.dynamic_rnn(decoder_cell, y, sequence_length=decoder_sequence_length, initial_state=encoder_final_states)

		W = tf.get_variable("W", shape=[None, self.config.decoder_hidden_size, self.config.vocab_size], initializer=tf.contrib.layers.xavier_initializer())
		b = tf.get_variable("b", shape=[None, self.config.max_sentence_len, self.config.vocab_size], initializer=tf.constant_initializer(0.0))

		preds = tf.matmul(outputs, W) + b

		return preds

	# Handles a single batch, returns the outputs
	def add_pred_single_batch_train(self):

		with tf.variable_scope(tf.get_variable_scope()):
			x = self.encoder_inputs_placeholder # must be 1D list of int32 Tensors of shape [batch_size]
			y = self.stacked_labels_placeholder

			# used encoder hidden size for output projection since this model uses a unidirectional LSTM encoder
			W = tf.get_variable("W", shape=[self.config.encoder_hidden_size, self.config.vocab_size], initializer=tf.contrib.layers.xavier_initializer())
			b = tf.get_variable("b", shape=[self.config.vocab_size], initializer=tf.constant_initializer(0.0))

			output_proj_vars = (W, b)

			x = tf.unstack(x, axis=0) # converts x into a list of tensors (list has max_sentence_len elems, each a tensor of shape [batch_size])
			y = tf.unstack(y, axis=0)

			lstm_cell = tf.contrib.rnn.LSTMCell(self.config.encoder_hidden_size, \
				initializer=tf.contrib.layers.xavier_initializer())
			preds, state = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(encoder_inputs=x, \
				decoder_inputs=y, cell=lstm_cell, num_encoder_symbols=self.config.vocab_size, \
				num_decoder_symbols=self.config.vocab_size, embedding_size=self.config.embed_size, num_heads=1, \
				output_projection=output_proj_vars, feed_previous=False, dtype=tf.float32)

			print("shape of preds returned by legacy:")
			print(preds[0].get_shape())

		return preds, W, b

	# Handles a single batch, returns the outputs
	def add_pred_single_batch_test(self, W, b):
		with tf.variable_scope(tf.get_variable_scope(), reuse=True):
			x = self.encoder_inputs_placeholder # must be 1D list of int32 Tensors of shape [batch_size]
			x = tf.unstack(x, axis=0) # converts x into a list of tensors (list has max_sentence_len elems, each a tensor of shape [batch_size])

			# don't have premade decoder inputs. will feed previous decoder output into next decoder cell's input
			output_proj_vars = (W, b)

			lstm_cell = tf.contrib.rnn.LSTMCell(self.config.encoder_hidden_size, \
				initializer=tf.contrib.layers.xavier_initializer())
			preds, state = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(encoder_inputs=x, \
				decoder_inputs=x, cell=lstm_cell, num_encoder_symbols=self.config.vocab_size, \
				num_decoder_symbols=self.config.vocab_size, embedding_size=self.config.embed_size, num_heads=1, \
				output_projection=output_proj_vars, \
				feed_previous=True, dtype=tf.float32)
		return preds

	# assumes we already have padding implemented.


	def add_loss_op(self, preds, W, b): # W and b refer to the weights and biases of the output projection matrix

		"""
		W: [encoder_hidden_size x vocab_size]
		b: [vocab_size]
		preds: [batch_size x max_sent_length x vocab_size]
		labels: [batch_size x max_sentence_length] (IDs. either convert self.stacked_labels_placeholder, or save original input)

		"""
		#labels = # need to fill this in with rank 2 tensor with words as ID numbers. can save in config

		projected_preds = [tf.matmul(pred, W) + b for pred in preds] # list, max_sentence_length long, of tensors: [batch_size x vocab_size]
		projected_preds = tf.stack(projected_preds, axis=1) # new shape: [batch_size, max_sentence_length, vocab_size]
		print("projected preds shape:")
		print(projected_preds.get_shape())

		print("labels shape:")
		print(self.unstacked_labels_placeholder.get_shape())

		ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.unstacked_labels_placeholder, logits=projected_preds)
		# shape of ce: same as labels, with same type as preds [batch_size x max_sentence_length]
		ce = tf.boolean_mask(ce, self.mask_placeholder)
		loss = tf.reduce_mean(ce)

		return loss


	def add_training_op(self, loss):

		train_op = tf.train.AdadeltaOptimizer(self.config.lr).minimize(loss) # same optimizer as in IBM paper
		# Similar to Adagrad, which gives smaller updates to frequent params and larger updates to infrequent parameters.
		# Improves on Adagrad by addressing Adagrad's aggressive, monotonically decreasing learning rate.

		return train_op

	# 
	def train_on_batch(self, sess, inputs_batch, labels_batch, mask_batch):
		feed = self.create_feed_dict(inputs_batch, \
			stacked_labels_batch=labels_batch, \
			mask_batch=mask_batch) # bug: mask_batch passed in has shape [max_sentence_length(20) x batch_size(3)], when it should be the opposite
		_, loss = sess.run([self.train_op, self.train_loss], feed_dict=feed)
		return loss


	def run_epoch(self, sess,  train_data, dev_data):
		prog = Progbar(target=1 + int(len(train_data) / self.config.batch_size))

		train_input_batches, train_truth_batches, train_mask_batches = train_data
		dev_input_batches, dev_truth_batches, dev_mask_batches = dev_data

		for i, input_batch in enumerate(train_input_batches):
			loss = self.train_on_batch(sess, input_batch, train_truth_batches[i], train_mask_batches[i])
			prog.update(i + 1, [("train loss", loss)])

			if self.report: self.report.log_train_loss(loss)
		print("")

		logger.info("Evaluating on development data")
		dev_loss = self.compute_dev_loss(sess, dev_input, dev_truth, dev_mask) # print loss on dev set

		return dev_loss # TODO: to check where the return value is used

	def fit(self, sess, saver):
		lowest_dev_loss = float("inf")

		#train_examples = self.preprocess_sequence_data(train_examples_raw)
		train_input, _, train_input_len = tokenize_data('train.ids.sentence', self.config.max_sentence_len, False)
		train_truth, train_truth_mask, train_truth_len = tokenize_data('train.ids.headline', self.config.max_sentence_len, True)

		#dev_set = self.preprocess_sequence_data(dev_set_raw)
		dev_input, _, dev_input_len = tokenize_data('val.ids.sentence', self.config.max_sentence_len, False)
		dev_truth, dev_truth_mask, dev_truth_len = tokenize_data('val.ids.headline', self.config.max_sentence_len, True)

		train_input_batches = get_stacked_minibatches(train_input, self.config.batch_size)
		train_truth_batches = get_stacked_minibatches(train_truth, self.config.batch_size)
		train_mask_batches = get_stacked_minibatches(train_truth_mask, self.config.batch_size)
	#	train_input_seq_len_batches = get_stacked_minibatches(train_input_len, self.config.batch_size)
	#	train_label_seq_len_batches = get_stacked_minibatches(train_truth_len, self.config.batch_size)

		dev_input_batches = get_stacked_minibatches(dev_input, self.config.batch_size)
		dev_truth_batches = get_stacked_minibatches(dev_truth, self.config.batch_size)
		dev_mask_batches = get_stacked_minibatches(dev_truth_mask, self.config.batch_size)
	#	dev_input_seq_len_batches = get_stacked_minibatches(dev_input_len, self.config.batch_size)
	#	dev_label_seq_len_batches = get_stacked_minibatches(dev_truth_len, self.config.batch_size)


		for epoch in range(self.config.n_epochs):
			logger.info("Epoch %d out of %d", epoch + 1, self.config.n_epochs)
			dev_loss = self.run_epoch(sess, (train_input_batches, train_truth_batches, train_mask_batches), \
										(dev_input_batches, dev_truth_batches, dev_mask_batches))
			if dev_loss < lowest_dev_loss:
				lowest_dev_loss = dev_loss
				if saver:
					logger.info("New lowest loss! Saving model in %s", self.config.model_output)
					saver.save(sess, self.config.model_output)
			print("")
			
			if self.report:
				self.report.log_epoch()
				self.report.save()
			
		return lowest_dev_loss

	def output_predictions(self, preds):
		output_file = open('dev_predictions', 'w')
		for i in range(tf.shape(preds)[0]):
			index = tf.argmax(preds[i])


	def build(self):
		self.add_placeholders()
		self.train_pred, W, b = self.add_pred_single_batch_train() # train_pred is stacked list (# elems = max_sent_length) of tensors: [batch_size x vocab_size]
		self.train_loss = self.add_loss_op(self.train_pred, W, b) # W and b refer to the weights and biases of the output projection matrix
		self.train_op = self.add_training_op(self.train_loss)

		self.dev_pred = self.add_pred_single_batch_test(W, b)
	#	output_predictions(self.dev_pred)
		self.dev_loss = self.add_loss_op(self.dev_pred, W, b)


	## Elliott's most recent additions

	# dev_loss is likely to be much higher than train_loss, since we're feeding in prev outputs (instead of ground truth)
	# into the decoder
	def compute_dev_loss(self, sess, inputs_batches, labels_batches, mask_batches):
		"""Compute dev loss for a single batch

		Args:
			sess: tf.Session()
			input_batch: np.ndarray of shape (n_samples, n_features)
		Returns:
			predictions: np.ndarray of shape (n_samples, n_classes)
		"""
		dev_loss = 0
		for i, input_batch in enumerate(inputs_batch):
			feed = self.create_feed_dict(input_batch, labels_batches[i], dev_mask[i])
			dev_loss += sess.run(self.dev_loss, feed_dict=feed)
		return dev_loss

'''
returns a list with lists containing the first words of all sentences, then the second words, then
the third words, etc. [[a1, b1, c1], [a2, b2, c2], [a3, b3, c3]] for sentences [a1, a2, a3], [b1, b2, b3] etc
'''
def get_stacked_minibatches(tokenized_data, batch_size):
	batches = []
	prev_val = 0
	for step in xrange(batch_size, len(tokenized_data) + batch_size, batch_size):
		batch = tokenized_data[prev_val:step]
		prev_val = step
		batches.append( np.stack(batch, axis=1) )
	return batches

def get_reg_minibatches(tokenized_data, batch_size):
	batches = []
	prev_val = 0
	for step in xrange(batch_size, len(tokenized_data) + batch_size, batch_size):
		batches.append( tokenized_data[prev_val:step] )
		prev_val = step
	return batches

def test_stacked_minibatches():
	data = [[1,2,3,4], [1,2,3,4], [1,2,3,4]]
	result = get_minibatches(data, 2)
	assert result == [[[1,1],[2,2],[3,3],[4,4]],[[1],[2],[3],[4]]]
	print("minibatches function is correct")

#Returns a list of sentences, which in turn are lists of integers that represent words
def tokenize_data(path, max_sentence_len, do_mask):
	tokenized_data = []
	masks = []
	sequence_length = []
	f = open('data/summarization/' + path,'r')
	for line in f.readlines():
		sentence = [int(x) for x in line.split()]
		if len(sentence) > max_sentence_len:
			continue
		sequence_length.append(len(sentence))
		if do_mask:
			mask = [True] * len(sentence)
			mask.extend([False] * (max_sentence_len - len(sentence)))
			masks.append(mask)
		sentence.extend([PAD_ID] * (max_sentence_len - len(sentence)))
		tokenized_data.append(sentence)
	print("Tokenized " + path)
	print(tokenized_data)
	return tokenized_data, masks, sequence_length

def do_train():

	# allows filehandler to write to the file specified by log_output
	config = Config()
	handler = logging.FileHandler(config.log_output)
	handler.setLevel(logging.DEBUG)
	handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
	logging.getLogger().addHandler(handler)

	with tf.Graph().as_default():
		logger.info("Building model...",)
		start = time.time()			
		rnn = RNN(config)
		logger.info("took %.2f seconds", time.time() - start)

		init = tf.global_variables_initializer() # saves an op to initialize variables

		saver = None

		with tf.Session() as session:
			session.run(init)
			rnn.fit(session, saver) # TODO: add spot for saver in fit. also need to pass in data to fit
			print("finished")
			# Save predictions in a text file.
	# 		output = model.output(session, dev_raw)
	#		sentences, labels, predictions = zip(*output)
	#		predictions = [[LBLS[l] for l in preds] for preds in predictions]
	#		output = zip(sentences, labels, predictions)

		#	with open(model.config.conll_output, 'w') as f:
		#		write_conll(f, output)
		#	with open(model.config.eval_output, 'w') as f:
		#		for sentence, labels, predictions in output:
		#			print_sentence(f, sentence, labels, predictions)


if __name__ == '__main__':
	do_train()


	'''	
	rnn.add_placeholders()
	rnn.create_feed_dict(input, ground_truth)
	rnn.encoder_decoder()
	'''

#GROUND TRUTH = HEADLINE
#INPUT = SENTENCE



'''
	def encoder(self):
		fwd_cell = tf.nn.tf.contrib.rnn.LSTMCell(encoder_hidden_size, initializer=tf.contrib.layers.xavier_initializer())
		bckwd_cell = tf.nn.tf.contrib.rnn.LSTMCell(encoder_hidden_size, initializer=tf.contrib.layers.xavier_initializer())
		x = self.inputs_placeholder
		outputs, output_states = tf.nn.bidirectional_dynamic_rnn(fwd_cell, bckwd_cell, x)
		return tf.concat(output_states, 2)

	def decoder(self, first_state):
		x = self.inputs_placeholder
		lstm_cell = tf.nn.tf.contrib.rnn.LSTMCell(decoder_hidden_size, initializer=tf.contrib.layers.xavier_initializer())
		
		tf.nn.seq2seq.attention_decoder(x, )

		tf.nn.dynamic_rnn(lstm_cell, x, initial_state=first_state)
'''




'''
def prepare_data():
	path = "data/summarization/"
	data_files = [path + "train.ids.headline", path + "train.ids.summary"]
	queue = tf.train.string_input_producer(data_files, num_epochs=10)  		
	reader = tf.TextLineReader()
	key, value = reader.read(queue)
	tensor = tf.decode_raw(value, tf.int32)
	


	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)
'''