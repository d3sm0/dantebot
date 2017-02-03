import tensorflow as tf
from tensorflow.python.ops import rnn_cell

class Model():

	
	def __init__(self, rnn_size = 158, num_layers = 3, learning_rate = 0.002, global_dropout = 0.98):

		self.rnn_size = rnn_size
		self.num_layers = num_layers

		self.learning_rate = learning_rate
		self.global_dropout = global_dropout
		

	def reset_graph():

		if 'sess' in globals() and sess:
			sess.close()
		tf.reset_default_graph()

	def build_graph(self, batch_size, seq_len, vocab_size):

		#Model.reset_graph()

		# input data
		self.x = tf.placeholder(tf.int32, [batch_size, seq_len])
		self.y = tf.placeholder(tf.int32, [batch_size, seq_len])

		# cells
		cell = rnn_cell.LSTMCell(self.rnn_size, state_is_tuple = True)
		#cell = rnn_cell.GRUCell(self.rnn_size)
		cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=self.global_dropout)
		
		# add state is tuple here
		cell = tf.nn.rnn_cell.MultiRNNCell([cell]*self.num_layers, state_is_tuple = True)
		self.cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.global_dropout)

		self.init = self.cell.zero_state(batch_size, tf.float32)

		# embeddings
		embeddings = tf.get_variable('embedding_matrix',[vocab_size, self.rnn_size])

		# array of batch_size * seq_len * rnn_size
		rnn_inputs = tf.nn.embedding_lookup(embeddings, self.x)

		# rnn_outputs is an array of batch_size * seq_len * rnn_size
		rnn_outputs, self.final_state = tf.nn.dynamic_rnn(self.cell, rnn_inputs, initial_state = self.init)

		tf.summary.histogram('last_state',self.final_state)

		# weight and bias declaration
		with tf.variable_scope('softmax') as scope:
			W = tf.get_variable('W',[self.rnn_size, vocab_size])
			b = tf.get_variable('b',[vocab_size], initializer=tf.constant_initializer(0.0))

		# reshape vectors
		# matrix of shape seq_len*batch_size, rnn_size
		self.rnn_outputs = tf.reshape(rnn_outputs, [-1, self.rnn_size])

		# array of len seq_len * batch_size
		y_ = tf.reshape(self.y, [-1])

		# logits
		self.logits = tf.matmul(self.rnn_outputs, W) + b
		tf.summary.histogram('logits',self.logits)

		# predictions
		self.predictions = tf.nn.softmax(self.logits)
		tf.summary.histogram('predictions',self.predictions)

		# cross entropy
		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits, y_)

		# loss
		self.loss = tf.reduce_mean(cross_entropy)
		tf.summary.scalar('loss',self.loss)

		# backprop
		self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
		
		tf.summary.scalar('learning_rate',self.learning_rate)

		
		self.summary_op = tf.summary.merge_all()

		return dict(
			x = self.x,
			y = self.y,
			init = self.init,
			final_state = self.final_state,
			loss = self.loss,
			train_step = self.train_step,
			predictions = self.predictions,
			saver = tf.train.Saver(),
			summary_op = self.summary_op
			)

	def variable_summaries(var):

		with tf.name_scope('summaries'):
			mean = tf.reduce_mean(var)
			tf.summary.scalar('mean',var)

			with tf.name_scope('sd'):
				sd = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
				tf.summary.scalar('sd', sd)
				tf.summary.scalar('max',tf.reduce_max(var))
				tf.summary.scalar('min',tf.reduce_min(var))
				tf.summary.histogram('histogram',var)




