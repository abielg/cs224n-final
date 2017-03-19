

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
SOS_ID = 1
UNK_ID = 2 

logger = logging.getLogger("final_project")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

model_path = ""

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
	decoder_hidden_size = encoder_hidden_size * 2
	batch_size = 10 # batch size was previously 2048
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

	def init_vocab(self):
		vocab_path = 'data/summarization/vocab.dat'
		if gfile.Exists(vocab_path):
			vocab = []
			with gfile.GFile(vocab_path, mode="r") as f:
				vocab.extend(f.readlines())
			vocab = [line.strip('\n') for line in vocab]
			return vocab

class RNN(object):
	def __init__(self, config):
		self.save_predictions = False
		self.config = config
		loaded = np.load('data/summarization/glove.trimmed.{}.npz'.format(str(self.config.embed_size)))
		self.embedding_matrix = loaded['glove']
		self.build()

	def __nonzero__(self):
		return self.save_predictions != 0

	def add_placeholders(self):
		self.encoder_inputs_placeholder = tf.placeholder(tf.int32, shape=([self.config.max_sentence_len, None]))
		self.stacked_labels_placeholder = tf.placeholder(tf.int32, shape=([self.config.max_sentence_len, None]))
		self.mask_placeholder = tf.placeholder(tf.bool, shape=([None, self.config.max_sentence_len])) # batch_sz x max_sentence_length
		self.sequence_placeholder = tf.placeholder(tf.int32, shape=([None]))

	def create_feed_dict(self, inputs_batch, unstacked_labels_batch=None, stacked_labels_batch=None, sequence_batch=None, mask_batch=None):
		feed_dict = {
			self.encoder_inputs_placeholder: inputs_batch,
		}

		if stacked_labels_batch is not None:
			feed_dict[self.stacked_labels_placeholder] = stacked_labels_batch

		if mask_batch is not None:
			feed_dict[self.mask_placeholder] = mask_batch
		if sequence_batch is not None:
			feed_dict[self.sequence_batch] = sequence_batch
		return feed_dict


	def add_embedding(self, placeholder):
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
		lookup_tensor = tf.nn.embedding_lookup(embedding_tensor, placeholder)
		#embeddings = tf.reshape(lookup_tensor, [-1, self.config.max_sentence_len, self.config.embed_size])
		print("Created embeddings tensor for the input")
		### END YOUR CODE
		return lookup_tensor

	# Handles a single batch, returns the outputs
	def add_pred_single_batch_train(self):

		with tf.variable_scope(tf.get_variable_scope()):
			#tensor of shape [max_sen_len, batch_size, embed_size]
			x = self.add_embedding(self.encoder_inputs_placeholder)
			
			#Tensor of shape [max_sen_len, batch_size, embed_size]
			y = self.add_embedding(self.stacked_labels_placeholder)

			#list of size max_sen_len with tensors of shape [batch_size, embed_size]
			y = tf.unstack(y, axis=0)

			sequence_length = self.sequence_placeholder

			fw_cell = tf.contrib.rnn.LSTMCell(self.config.encoder_hidden_size, initializer=tf.contrib.layers.xavier_initializer())
			bckwd_cell = tf.contrib.rnn.LSTMCell(self.config.encoder_hidden_size, initializer=tf.contrib.layers.xavier_initializer())

			encoder_outputs, encoder_final_states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bckwd_cell, x, sequence_length=sequence_length, time_major=True, dtype=tf.float64)			

			final_state_fw, final_state_bw = encoder_final_states

			encoder_final_states = tf.concat([final_state_fw[1], final_state_bw[1]], 1)


			encoder_outputs = (tf.transpose(encoder_outputs[0], (1,0,2)), tf.transpose(encoder_outputs[1], (1,0,2)))
			attention_states = tf.concat(encoder_outputs, 2)

			decoder_cell = tf.contrib.rnn.LSTMCell(self.config.decoder_hidden_size, \
							initializer=tf.contrib.layers.xavier_initializer(), state_is_tuple=False)
			
			#SETTING NUM_HEADS TO 8
			decoder_outputs, decoder_state = tf.contrib.legacy_seq2seq.attention_decoder(y, encoder_final_states, attention_states,\
											 decoder_cell, output_size=self.config.vocab_size, num_heads=8, dtype=tf.float64)

			#decoder_outputs, decoder_state = tf.contrib.legacy_seq2seq.rnn_decoder(y, encoder_final_states, decoder_cell)

			# used encoder hidden size for output projection since this model uses a unidirectional LSTM encoder
			#W = tf.get_variable("W", shape=[self.config.encoder_hidden_size, self.config.vocab_size], initializer=tf.contrib.layers.xavier_initializer())
			#b = tf.get_variable("b", shape=[self.config.vocab_size], initializer=tf.constant_initializer(0.0))

			#decoder_ouputs is a list of max_sen_len elems with tensors of size (batch, vocab_size)

		return decoder_outputs

	# Handles a single batch, returns the outputs
	def add_pred_single_batch_test(self):
		with tf.variable_scope(tf.get_variable_scope(), reuse=True):
			#tensor of shape [max_sen_len, batch_size, embed_size]
			x = self.add_embedding(self.encoder_inputs_placeholder)

			sequence_length = self.sequence_placeholder

			fw_cell = tf.contrib.rnn.LSTMCell(self.config.encoder_hidden_size, initializer=tf.contrib.layers.xavier_initializer())
			bckwd_cell = tf.contrib.rnn.LSTMCell(self.config.encoder_hidden_size, initializer=tf.contrib.layers.xavier_initializer())

			encoder_outputs, encoder_final_states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bckwd_cell, x, sequence_length=sequence_length, time_major=True, dtype=tf.float64)			

			final_state_fw, final_state_bw = encoder_final_states

			encoder_final_states = tf.concat([final_state_fw[1], final_state_bw[1]], 1)

			encoder_outputs = (tf.transpose(encoder_outputs[0], (1,0,2)), tf.transpose(encoder_outputs[1], (1,0,2)))

			attention_states = tf.concat(encoder_outputs, 2)

			decoder_cell = tf.contrib.rnn.LSTMCell(self.config.decoder_hidden_size, \
							initializer=tf.contrib.layers.xavier_initializer(), state_is_tuple=False)


			#loop function: receives an output of dim vocab_size, does arg_max, and gets that word's embedding
			'''
			def loop_function(prev, i):
				preds = tf.argmax(prev, axis=1)
				input = self.add_embedding(preds)
				return input
			'''
			#SETTING NUM_HEADS TO 8
			decoder_outputs, decoder_state = tf.contrib.legacy_seq2seq.attention_decoder(tf.unstack(x), encoder_final_states, attention_states,\
											 decoder_cell, output_size=self.config.vocab_size, num_heads=8,\
											 dtype=tf.float64)


		return decoder_outputs
	# assumes we already have padding implemented.

	def add_loss_op(self, preds, print_pred=False): # W and b refer to the weights and biases of the output projection matrix

		"""
		W: [encoder_hidden_size x vocab_size]
		b: [vocab_size]
		preds: [batch_size x max_sent_length x vocab_size]
		labels: [batch_size x max_sentence_length] (IDs. either convert self.stacked_labels_placeholder, or save original input)

		"""
		#labels = # need to fill this in with rank 2 tensor with words as ID numbers. can save in config

		#projected_preds = [tf.matmul(pred, W) + b for pred in preds] # list, max_sentence_length long, of tensors: [batch_size x vocab_size]
		
		#by stacking, preds will be a tensor of shape (max_sentence_len, batch_size, vocab_size)
		preds = tf.stack(preds)

		preds = tf.transpose(preds, (1,0,2)) # new shape: [batch_size, max_sentence_length, vocab_size]

		unstacked_labels = tf.transpose(self.stacked_labels_placeholder) # shape: [batch_size x max_sentence_len]

		ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=unstacked_labels, logits=preds)
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
	def train_on_batch(self, sess, inputs_batch, labels_batch, mask_batch, sequence_batch):
		feed = self.create_feed_dict(inputs_batch=inputs_batch, \
			stacked_labels_batch=labels_batch, \
			mask_batch=mask_batch, \
			sequence_batch=sequence_batch)
		_, loss = sess.run([self.train_op, self.train_loss], feed_dict=feed)
		return loss


	def predict_on_batch(self, sess, inputs_batch, labels_batch, mask_batch, num_of_batch=2, using_dev=True):
		feed = self.create_feed_dict(inputs_batch=inputs_batch, \
			stacked_labels_batch=labels_batch, \
			mask_batch=mask_batch)

		preds = None # make sure this is legit
		loss = None

		if (using_dev):
			preds, loss = sess.run([self.dev_pred, self.dev_loss], feed)
		else:
			preds, loss = sess.run([self.test_pred, self.test_loss], feed)

		if (self.save_predictions == True):
			if num_of_batch % 2 == 0:
				self.save_outputs(sess, preds, inputs_batch, labels_batch, num_preds=1)

		return loss

	def save_outputs(self, sess, preds, inputs, titles, num_preds=-1): # shape of each input: [batch_size x max_sentence_length]
		preds = tf.stack(preds, axis=1) # new shape: [batch_size, max_sentence_length, vocab_size]
		preds = tf.argmax(preds, axis=2) # new shape: [batch_size, max_sentence_length]

	#	inputs = self.encoder_inputs_placeholder # shape: [max_sentence_len, batch_size]
	#	inputs = tf.unstack(inputs, axis=0) 
		inputs = tf.transpose(inputs) # new shape: [batch_size, max_sentence_len]

		titles = tf.transpose(titles)

		inputs_list = tf.unstack(inputs, num=self.config.batch_size) # batch_size elems, each a tensor: [max_sentence_len]
		titles_list = tf.unstack(titles, num=self.config.batch_size)
		preds_list =tf.unstack(preds, num=self.config.batch_size)

		with gfile.GFile(self.config.preds_output, mode="a") as output_file:
			logger.info("Storing predictions in " + self.config.preds_output)
			for i, _input in enumerate(inputs_list[:num_preds]):
				
				_input = _input.eval(session=sess)
				title = titles_list[i].eval(session=sess)
				pred = preds_list[i].eval(session=sess)

				output_file.write("article (input): ")
				for index in tf.unstack(_input): # input is a numpy array. iterate through it somehow
					w = self.config.vocabulary[index.eval(session=sess)]
					output_file.write(w + " ")
				output_file.write("\n")

				output_file.write("prediction: ")
				for index in tf.unstack(pred):
					w = self.config.vocabulary[index.eval(session=sess)]
					output_file.write(w + " ")
				output_file.write("\n")

				output_file.write("title (truth): ")
				for index in tf.unstack(title):
					w = self.config.vocabulary[index.eval(session=sess)]
					output_file.write(w + " ")
				output_file.write("\n \n")

	# dev_loss is likely to be much higher than train_loss, since we're feeding in prev outputs (instead of ground truth)
	# into the decoder
	def compute_dev_loss(self, sess, input_batches, labels_batches, mask_batches):

		prog = Progbar(target=1 + len(input_batches))
		total_dev_loss = 0
		for i, input_batch in enumerate(input_batches):

		#	feed = self.create_feed_dict(inputs_batch=input_batch, stacked_labels_batch=labels_batches[i], mask_batch=mask_batches[i]) #problem: labels has shape: [batch_size x max_sentence_length], should be opposite
		#	dev_loss = sess.run(self.dev_loss, feed_dict=feed)
			dev_loss = self.predict_on_batch(sess, inputs_batch=input_batch, labels_batch=labels_batches[i], \
				mask_batch=mask_batches[i], num_of_batch=i, using_dev=True)

			total_dev_loss += dev_loss
			prog.update(i + 1, [("dev loss", dev_loss)])
			if i == len(input_batches) - 1:
				logger.info("Last batch dev loss: " + str(dev_loss))
		return total_dev_loss


	def run_epoch(self, sess,  train_data, dev_data):
		train_input_batches, train_truth_batches, train_mask_batches, train_input_sequence = train_data
		dev_input_batches, dev_truth_batches, dev_mask_batches, dev_input_sequence = dev_data

		logger.info("number of train input batches: %d", int(len(train_input_batches)))
		prog = Progbar(target=1 + len(train_input_batches))

		loss = 0
		for i, input_batch in enumerate(train_input_batches):
			loss = self.train_on_batch(sess, input_batch, train_truth_batches[i], train_mask_batches[i], train_input_sequence[i])
			prog.update(i + 1, [("train loss", loss)])
		logger.info("\nTrain loss: " + str(loss))

		#	if self.report: self.report.log_train_loss(loss)
		#print("")

		dev_loss = self.compute_dev_loss(sess, dev_input_batches, dev_truth_batches, dev_mask_batches) # print loss on dev set

		return dev_loss # TODO: to check where the return value is used


	# function called when working with test set. outputs loss on test set, along with the model's predictions
	def preds_and_loss(self, sess, saver): # not sure which of these params we actually need
		# TODO: make sure what we're working with is actually 'test.ids.article'
		test_input, _, test_input_len = tokenize_data('test.ids.article', self.config.max_sentence_len, False)
		test_truth, test_truth_mask, test_truth_len = tokenize_data('test.ids.title', self.config.max_sentence_len, True)

		test_input_batches = get_stacked_minibatches(test_input, self.config.batch_size)
		test_truth_batches = get_stacked_minibatches(test_truth, self.config.batch_size)
		test_mask_batches = get_reg_minibatches(test_truth_mask, self.config.batch_size)

		# run through once (don't need multiple epochs)

		prog = Progbar(target=1 + int(len(test_input_batches) / self.config.batch_size))

		total_test_loss = 0
		self.save_predictions = True
		for i, input_batch in enumerate(test_input_batches):
			loss = self.predict_on_batch(sess, input_batch, test_truth_batches[i], test_mask_batches[i], num_of_batch=i, using_dev=False)
			total_test_loss += loss
			prog.update(i + 1, [("test loss on batch", loss)])

		return total_test_loss

	def fit(self, sess, saver):
		lowest_dev_loss = float("inf")

		train_input, _, train_input_len = tokenize_data('train.ids.article', self.config.max_sentence_len, False)
		train_truth, train_truth_mask, train_truth_len = tokenize_data('train.ids.title', self.config.max_sentence_len, True)

		dev_input, _, dev_input_len = tokenize_data('val.ids.article', self.config.max_sentence_len, False)
		dev_truth, dev_truth_mask, dev_truth_len = tokenize_data('val.ids.title', self.config.max_sentence_len, True)

		train_input_batches = get_stacked_minibatches(train_input, self.config.batch_size)
		train_truth_batches = get_stacked_minibatches(train_truth, self.config.batch_size)
		train_mask_batches = get_reg_minibatches(train_truth_mask, self.config.batch_size)
		train_input_sequence = get_reg_minibatches(train_input_len, self.config.batch_size)

		logger.info("number of training input batches, as indicated by fit: %d", len(train_input_batches))

		dev_input_batches = get_stacked_minibatches(dev_input, self.config.batch_size)
		dev_truth_batches = get_stacked_minibatches(dev_truth, self.config.batch_size)
		dev_mask_batches = get_reg_minibatches(dev_truth_mask, self.config.batch_size)
		dev_input_sequence = get_reg_minibatches(dev_input_len, self.config.batch_size)

		for epoch in range(self.config.n_epochs):
			if epoch == self.config.n_epochs - 2:
				self.save_predictions = True
			else:
				self.save_predictions = False

			logger.info("Epoch %d out of %d", epoch + 1, self.config.n_epochs)
			dev_loss = self.run_epoch(sess, (train_input_batches, train_truth_batches, train_mask_batches, train_input_sequence), \
										(dev_input_batches, dev_truth_batches, dev_mask_batches, dev_input_sequence))
			logger.info("Epoch #%d dev loss: %d", epoch + 1, dev_loss)
			if dev_loss < lowest_dev_loss:
				lowest_dev_loss = dev_loss
				if saver:
					logger.info("New lowest loss! Saving model in %s", self.config.model_output)
					saver.save(sess, self.config.model_output) # saves parameters for best-performing model (lowest total dev loss)
			
			print("")
			
	#		if self.report:
	#			self.report.log_epoch()
	#			self.report.save()
			
		return lowest_dev_loss


	def output_predictions(self, preds):
		output_file = open('dev_predictions', 'w')
		for i in range(tf.shape(preds)[0]):
			index = tf.argmax(preds[i])


	def build(self):
		self.add_placeholders()
		self.train_pred = self.add_pred_single_batch_train() # train_pred is stacked list (# elems = max_sent_length) of tensors: [batch_size x vocab_size]
		self.train_loss = self.add_loss_op(self.train_pred) # W and b refer to the weights and biases of the output projection matrix
		self.train_op = self.add_training_op(self.train_loss)

		self.dev_pred = self.add_pred_single_batch_test()
		self.dev_loss = self.add_loss_op(self.dev_pred)

		self.test_pred = self.add_pred_single_batch_test()
		self.test_loss = self.add_loss_op(self.test_pred)

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

#Returns a list of sentences, which in turn are lists of integers that represent words
def tokenize_data(path, max_sentence_len, do_mask):
	tokenized_data = []
	masks = []
	sequence_length = []
	f = open('data/summarization/' + path,'r')
	for line in f.readlines():
		sentence = [int(x) for x in line.split()]
		if len(sentence) > max_sentence_len:
			sentence = sentence[:max_sentence_len]
		sequence_length.append(len(sentence))
		if do_mask:
			mask = [True] * len(sentence)
			mask.extend([False] * (max_sentence_len - len(sentence)))
			masks.append(mask)
		sentence.extend([PAD_ID] * (max_sentence_len - len(sentence)))
		tokenized_data.append(sentence)
	print("Tokenized " + path + " with %d sentences" % len(tokenized_data))
	return tokenized_data, masks, sequence_length

def do_train(args):

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

		saver = tf.train.Saver() # adds ops to save and restore variables to and from checkpoints

		with tf.Session() as session:
			session.run(init)
			rnn.fit(session, saver)
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

# using previously trained parameters, calculates loss and generates labels for unseen data
def do_test(args):

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

		saver = tf.train.Saver()

		with tf.Session() as session:
			session.run(init)
			print("Applying saved params: " + args.saved_params)
			saver.restore(session, args.saved_params) # restores and initiales old saved params. make sure this works
			# TODO: create way of inputting model_output params that we want to evaluate on

			# TODO: need method of taking in input_data
			total_loss = rnn.preds_and_loss(session, saver)
			logger.info("Total loss on test set: %d", total_loss)
			# get outputs on data

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Trains and tests an attentive model for abstractive summarization')
	subparsers = parser.add_subparsers()

	command_parser = subparsers.add_parser('train', help='')
	command_parser.set_defaults(func=do_train)

	command_parser = subparsers.add_parser('test', help='')
	parser.add_argument("--saved_params")
	#command_parser.add_argument('-p', '--saved-params', type=argparse.FileType('r'), help="Saved params to use when testing")
	command_parser.set_defaults(func=do_test)

	ARGS = parser.parse_args()
	if ARGS.func is None:
		parser.print_help()
		sys.exit(1)
	else:
		ARGS.func(ARGS)
		model_path = ARGS.saved_params