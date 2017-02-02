import tensorflow as tf
from tensorflow.python.ops import rnn_cell

class Model():
	def __init__(self,args):

		if args == None:
			self.args = Args()

	def build_graph(self, args):

		# input data
		self.x = tf.placeholder(tf.int32,[args.batch_size, args.seq_len])
		self.y = tf.placeholder(tf.int32,[args.batch_size, args.seq_len])

		# cells 
		cell = rnn_cell.GRUCell(args.rnn_size)
		cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=args.global_dropout)
		cell = tf.nn.rnn_cell.MultiRNNCell([cell]*args.num_layers)
		self.cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=args.global_dropout)

		self.init = cell.zero_state(args.batch_size, tf.float32)

		# embeddings
		embeddings = tf.get_variable('embedding_matrix',[args.vocab_size, args.rnn_size])

		# array of batch_size * seq_len * rnn_size
		rnn_inputs = tf.nn.embedding_lookup(embeddings, self.x)

		# rnn_outputs is an array of batch_size * seq_len * rnn_size
		rnn_outputs, self.final_state = tf.nn.dynamic_rnn(self.cell, rnn_inputs, initial_state = self.init)

		tf.summary.histogram('last_state',self.final_state)

		# weight and bias declaration
		with tf.variable_scope('softmax') as scope:
			W = tf.get_variable('W',[args.rnn_size, args.vocab_size])
			b = tf.get_variable('b',[args.vocab_size], initializer=tf.constant_initializer(0.0))

		# reshape vectors
		# matrix of shape seq_len*batch_size, rnn_size
		self.rnn_outputs = tf.reshape(rnn_outputs, [-1, args.rnn_size])

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
		self.train_step = tf.train.AdamOptimizer(args.learning_rate).minimize(self.loss)

		# save current state
		self.saver = tf.train.Saver()
		
		# merge summaries
		self.summary_op = tf.summary.merge_all()

		return dict(
			x = self.x,
			y = self.y,
			init = self.init,
			final_state = self.final_state,
			loss = self.loss,
			train_step = self.train_step,
			predictions = self.predictions,
			saver = self.saver
			)




