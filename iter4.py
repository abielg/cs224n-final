

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
	dropout = 0.5
	embed_size = 100
	encoder_hidden_size = 200
	decoder_hidden_size = encoder_hidden_size
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
		self.encoder_inputs_placeholder = tf.placeholder(tf.int32, shape=([None, self.config.max_sentence_len]))
		self.labels_placeholder = tf.placeholder(tf.int32, shape=([None, self.config.max_sentence_len]))
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
			feed_dict[self.sequence_placeholder] = sequence_batch
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
		### END YOUR CODE
		return lookup_tensor

	# Handles a single batch, returns the outputs
	def add_pred_single_batch_train(self):
		with tf.variable_scope(tf.get_variable_scope()):
			input_sequence_lengths = self.sequence_placeholder
			encoder_outputs, final_enc_state = self.encode() # TODO: gather sequence lengths
			preds = self.decode_train(encoder_outputs, final_enc_state)

		return preds # [batch_size, max_sentence_len, vocab_size]

	def encode(self):

		#inputs: [batch_size x max_sentence_length x embed_size]
		#sequence_length: [batch_size] vector containing actual length for each sequence

		# encoder_hidden_size is considered the size of the concatenation of the forward and backward cells
		input_sequence_lengths = self.sequence_placeholder # [batch_size] tensor of the lengths of each input sentence in the batch
		x = self.add_embedding(self.encoder_inputs_placeholder) # [batch_sz, max_sentence_len, embed_sz]

	#	fw_cell2 = tf.nn.rnn_cell.LSTMCell(num_units=.5 * self.config.encoder_hidden_size, initializer=tf.contrib.layers.xavier_initializer())
	#	bckwd_cell2 = tf.nn.rnn_cell.LSTMCell(num_units=.5 * self.config.encoder_hidden_size, initializer=tf.contrib.layers.xavier_initializer())

		fw_cell = tf.contrib.rnn.LSTMCell(.5 * self.config.encoder_hidden_size, initializer=tf.contrib.layers.xavier_initializer())
		bckwd_cell = tf.contrib.rnn.LSTMCell(.5 * self.config.encoder_hidden_size, initializer=tf.contrib.layers.xavier_initializer())

		outputs, final_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell, cell_bw=bckwd_cell, \
			inputs=x, sequence_length=input_sequence_lengths, dtype=tf.float64)

	#	outputs_fw, outputs_bw = outputs
	#	outputs_fw = tf.cast(outputs_fw, tf.float32)
	#	outputs_bw = tf.cast(outputs_bw, tf.float32)

		concat_outputs = tf.concat(outputs, 2) # dimension: [batch_size x max_sentence_length x encoder_hidden_size]

		final_state_fw, final_state_bw = final_state

		final_state = tf.concat([final_state_fw[1], final_state_bw[1]], 1)

		return concat_outputs, final_state # concat_outputs -> attention_values, sequence_length -> attention_values_length
		# concat_outputs: [batch_size, max_sentence_len, output_size (encoder_hidden_size)]


	def decode_train(self, attention_values, decoder_start_state):

		input_embeddings = self.add_embedding(self.labels_placeholder) # truth embeddings. [batch_size x max_sentence_length x embed_size]

		preds = [] # currently saved as a list, where each elem represents one timestep. TODO: reshape to tensor, where max_sentence_len is second dimension

		cur_hidden_state = None
		cur_inputs = None

		with tf.variable_scope("lstm") as scope:

			for i in xrange(self.config.max_sentence_len + 1): # max_sequence_length iterations + 1 (+1 for start symbol). starts at 0
				if i > 0:
					scope.reuse_variables()

				if (i==0): 
					cur_hidden_state = decoder_start_state   # shape: [batch_size, decoder_hidden_size]
					start_symbol_tensor = tf.constant(value=SOS_ID, dtype=tf.int32, shape=[self.config.batch_size])
					cur_inputs = self.add_embedding(start_symbol_tensor) # TODO: fairly unsure whether or not this will work. will just try
				else:
					# cur_hidden_state initialized previously at end of for loop
				#	cur_inputs = input_embeddings[:, i-1, :] # desired shape: [batch_size, embed_size]. unsure whether or not this will work (might return as numpy array?)
					cur_inputs = tf.slice(input_embeddings, [0, i-1, 0], [-1, 1, -1])

				intermediate_hidden_state = None
				if (i==0): # state_is_tuple is true for first cell bc end of encoder function outputs final state in weird ass format
	
					lstmCell = tf.contrib.rnn.LSTMCell(num_units=self.config.decoder_hidden_size, \
						initializer=tf.contrib.layers.xavier_initializer(), state_is_tuple=False)
					intermediate_hidden_state, _ = lstmCell(cur_inputs, cur_hidden_state) # LSTM output has shape: [batch_size, decoder_hidden_size]

				else:
					lstmCell = tf.contrib.rnn.LSTMCell(num_units=self.config.decoder_hidden_size, \
						initializer=tf.contrib.layers.xavier_initializer(), state_is_tuple=False)
					intermediate_hidden_state, _ = lstmCell(cur_inputs, cur_hidden_state) # LSTM output has shape: [batch_size, decoder_hidden_size]

				if (i==0):
					# layer_inputs = intermediate_hidden_state
					next_hidden_state = intermediate_hidden_state
				else:
					attn_scores, context_vector = self.attention_function_simple(intermediate_hidden_state, attention_values, self.sequence_placeholder)
					
					layer_inputs = tf.concat([intermediate_hidden_state, context_vector], 1) # size: [batch_size, 2 * decoder_hidden_size]
					next_hidden_state = tf.contrib.layers.fully_connected(inputs=layer_inputs, \
						num_outputs=self.config.decoder_hidden_size, activation_fn=tf.nn.tanh, trainable=True)  # should be trainable as is

					W_out = tf.get_variable("W_out", shape=[self.config.decoder_hidden_size, self.config.vocab_size], initializer=tf.contrib.layers.xavier_initializer())
					b_out = tf.get_variable("b_out", shape=[self.config.vocab_size], initializer=tf.constant_initializer(0.0))

					cur_pred = tf.matmul(cur_hidden_state, W_out) + b_out
					preds.append(cur_pred)

				cur_hidden_state = next_hidden_state
				print("made it through first iteration of for loop")

		preds_tensor_form = tf.stack(preds, axis=1) # axis = 1 so that max_sentence_len will be second dimension

		return preds_tensor_form # TODO: EOS token? or output one pad token -> fill the rest with pad?

	def attention_function_simple(self, cell_output, attention_values, sequence_length): # h{enc_i} dot h{s} ... h{s} represents previous hidden state of decoder

		# cell_output: [batch_size, decoder_hidden_size] (same dimension as concat_outputs) # h(t-1) of decoder
		# attention_values: [batch_size, max_sentence_len, encoder_hidden_size] ... okay bc we're using global attention

	#	W = tf.get_variable(shape=[self.encoder_hidden_size, self.decoder_hidden_size], initializer=tf.contrib.layers.xavier_initializer()) # manually set scope?

	#	first_dot = tf.matmul(attention_values, W) # [batch_size, max_sentence_len, 2 * encoder_hidden_size]
	#	print("shape of first dot: ")
	#	print(first_dot.get_shape)

		# global attention. using bilinear formula: ht dot W dot cell_output
		raw_scores = tf.reduce_sum(attention_values * tf.expand_dims(cell_output, 1), [2]) # [batch_sz, max_sentence_len]
		scores_mask = tf.sequence_mask(lengths=tf.to_int32(sequence_length), maxlen=self.config.max_sentence_len, dtype=tf.float64)
		attn_scores = raw_scores * scores_mask + ((1.0 - scores_mask) * tf.float64.min) # unsure what exactly this does ... seems to be a typecast
		prob_distribution = tf.nn.softmax(attn_scores) # [batch_sz, max_sentence_len]
		context_vector = tf.expand_dims(prob_distribution, 2) * attention_values # multiplies each hidden encoder vector by its attention score
		# context_vectors shape: [batch_sz, max_sentence_len, dec_hidden_size]
		context_vector = tf.reduce_sum(context_vector, 1) # reduced over max_sentence_len dimension bc we're using global attention
		# unsure if we actually need context.set_shape ... shape should be: [batch_size, dec_hidden_size]

		return (prob_distribution, context_vector)


	# Handles a single batch, returns the outputs
	def add_pred_single_batch_test(self, W, b):
		# TODO: 
		return

	# assumes we already have padding implemented.

	def add_loss_op(self, preds, print_pred=False):

		"""
		preds: [batch_size x max_sent_length x vocab_size]
		labels: [batch_size x max_sentence_length] (IDs)

		"""
		labels = self.labels_placeholder
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
			sequence_batch=sequence_batch) # TODO: what does sequence_batch mean?
		_, loss = sess.run([self.train_op, self.train_loss], feed_dict=feed)
		return loss


	def predict_on_batch(self, sess, inputs_batch, labels_batch, mask_batch, sequence_batch, num_of_batch=2, using_dev=True):
		feed = self.create_feed_dict(inputs_batch=inputs_batch, \
			stacked_labels_batch=labels_batch, \
			mask_batch=mask_batch, \
			sequence_batch=sequence_batch)

		preds = None # make sure this is legit
		loss = None

		if (using_dev):
			preds, loss = sess.run([self.dev_pred, self.dev_loss], feed)
		else:
			preds, loss = sess.run([self.test_pred, self.test_loss], feed)

		if (self.save_predictions == True):
			if num_of_batch % 40 == 0:
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
	def compute_dev_loss(self, sess, input_batches, labels_batches, mask_batches, sequence_batches):

		prog = Progbar(target=1 + len(input_batches))
		total_dev_loss = 0
		for i, input_batch in enumerate(input_batches):

		#	feed = self.create_feed_dict(inputs_batch=input_batch, stacked_labels_batch=labels_batches[i], mask_batch=mask_batches[i]) #problem: labels has shape: [batch_size x max_sentence_length], should be opposite
		#	dev_loss = sess.run(self.dev_loss, feed_dict=feed)
			dev_loss = self.predict_on_batch(sess, inputs_batch=input_batch, labels_batch=labels_batches[i], \
				mask_batch=mask_batches[i], sequence_batch=sequence_batches[i], num_of_batch=i, using_dev=True)

			total_dev_loss += dev_loss
			prog.update(i + 1, [("dev loss", dev_loss)])
			if i == len(input_batches) - 1:
				logger.info("Last batch dev loss: %d", dev_loss)
		return total_dev_loss


	def run_epoch(self, sess, train_data, dev_data):
		train_input_batches, train_truth_batches, train_mask_batches, train_input_sequence = train_data
		dev_input_batches, dev_truth_batches, dev_mask_batches, dev_input_sequence = dev_data

		logger.info("number of train input batches: %d", int(len(train_input_batches)))
		prog = Progbar(target=1 + len(train_input_batches))

		loss = 0
		for i, input_batch in enumerate(train_input_batches):
			loss = self.train_on_batch(sess, input_batch, train_truth_batches[i], train_mask_batches[i], train_input_sequence[i])
			prog.update(i + 1, [("train loss", loss)])
		logger.info("\nTrain loss: %d", loss)

		#	if self.report: self.report.log_train_loss(loss)
		#print("")

		dev_loss = self.compute_dev_loss(sess, dev_input_batches, dev_truth_batches, dev_mask_batches, train_input_sequence) # print loss on dev set

		return dev_loss # TODO: to check where the return value is used


	# function called when working with test set. outputs loss on test set, along with the model's predictions
	def preds_and_loss(self, sess, saver): # not sure which of these params we actually need
		# TODO: make sure what we're working with is actually 'test.ids.article'
		test_input, _, test_input_len = tokenize_data('test.ids.article', self.config.max_sentence_len, False)
		test_truth, test_truth_mask, test_truth_len = tokenize_data('test.ids.title', self.config.max_sentence_len, True)

		test_input_batches = get_stacked_minibatches(test_input, self.config.batch_size)
		test_truth_batches = get_stacked_minibatches(test_truth, self.config.batch_size)
		test_mask_batches = get_reg_minibatches(test_truth_mask, self.config.batch_size)
		test_input_sequence_batches = get_reg_minibatches(test_input_len, self.config.batch_size)

		# run through once (don't need multiple epochs)

		prog = Progbar(target=1 + int(len(test_input_batches) / self.config.batch_size))

		total_test_loss = 0
		self.save_predictions = True
		for i, input_batch in enumerate(test_input_batches):
			loss = self.predict_on_batch(sess, input_batch, test_truth_batches[i], test_mask_batches[i],\
										test_input_sequence_batches[i], num_of_batch=i, using_dev=False)
			total_test_loss += loss
			prog.update(i + 1, [("test loss on batch", loss)])

		return total_test_loss

	def fit(self, sess, saver):
		lowest_dev_loss = float("inf")

		train_input, _, train_input_len = tokenize_data('train.ids.article', self.config.max_sentence_len, False)
		train_truth, train_truth_mask, _ = tokenize_data('train.ids.title', self.config.max_sentence_len, True)

		dev_input, _, dev_input_len = tokenize_data('val.ids.article', self.config.max_sentence_len, False)
		dev_truth, dev_truth_mask, _ = tokenize_data('val.ids.title', self.config.max_sentence_len, True)

		train_input_batches = get_reg_minibatches(train_input, self.config.batch_size)
		train_truth_batches = get_reg_minibatches(train_truth, self.config.batch_size)
		train_mask_batches = get_reg_minibatches(train_truth_mask, self.config.batch_size)
		train_input_sequence = get_reg_minibatches(train_input_len, self.config.batch_size)

		logger.info("number of training input batches, as indicated by fit: %d", len(train_input_batches))

		dev_input_batches = get_reg_minibatches(dev_input, self.config.batch_size)
		dev_truth_batches = get_reg_minibatches(dev_truth, self.config.batch_size)
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
		self.train_pred= self.add_pred_single_batch_train()
		self.train_loss = self.add_loss_op(self.train_pred)
		self.train_op = self.add_training_op(self.train_loss)

		self.dev_pred = self.add_pred_single_batch_test(W, b)
		self.dev_loss = self.add_loss_op(self.dev_pred, W, b)

		self.test_pred = self.add_pred_single_batch_test(W, b)
		self.test_loss = self.add_loss_op(self.test_pred, W, b)

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
		if path[-5:] == 'title':
			sentence.insert(0,SOS_ID)
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
		model = RNN(config)
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