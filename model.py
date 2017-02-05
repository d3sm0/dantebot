import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import seq2seq

class Model():

	
	def __init__(self, rnn_size = 158, num_layers = 3, learning_rate = 0.001, global_dropout = 0.9, grad_clip = 5.):

		self.rnn_size = rnn_size
		self.num_layers = num_layers

		self.learning_rate = learning_rate
		self.global_dropout = global_dropout
		self.grad_clip = grad_clip
		

	def reset_graph():

		if 'sess' in globals() and sess:
			sess.close()
		tf.reset_default_graph()

	def build_graph(self, batch_size, seq_len, vocab_size, infer = False):

		# input data
		self.x = tf.placeholder(tf.int32, [batch_size, seq_len])
		self.y = tf.placeholder(tf.int32, [batch_size, seq_len])

		# cells
		cell = rnn_cell.LSTMCell(self.rnn_size, state_is_tuple = True, activation = tf.nn.elu)
		cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=self.global_dropout)
		cell = tf.nn.rnn_cell.MultiRNNCell([cell]*self.num_layers, state_is_tuple = True)

		self.cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.global_dropout)
		self.init = self.cell.zero_state(batch_size, tf.float32)
			
		# embeddings
		embeddings = tf.get_variable('embedding_matrix',[vocab_size, self.rnn_size])

		# array of batch_size * seq_len * rnn_size
		rnn_inputs = tf.nn.embedding_lookup(embeddings, self.x)
		
		# rnn_outputs is an array of batch_size * seq_len * rnn_size
		rnn_outputs, self.final_state = tf.nn.dynamic_rnn(self.cell, rnn_inputs, initial_state = self.init)
	
		# matrix of shape seq_len*batch_size, rnn_size
		self.rnn_outputs = tf.reshape(tf.concat(1,rnn_outputs),[-1, self.rnn_size])

		# weight and bias declaration
		W = tf.get_variable('W',[self.rnn_size, vocab_size])
		b = tf.get_variable('b',[vocab_size], initializer=tf.constant_initializer(0.0))
		
		# logits
		self.logits = tf.matmul(self.rnn_outputs, W) + b
	
		# predictions
		self.probs = tf.nn.softmax(self.logits)
		
		# array of len seq_len * batch_size
		y_ = tf.reshape(self.y, [-1])

		# cross entropy
		cross_entropy = seq2seq.sequence_loss_by_example([self.logits], [y_], [tf.ones([batch_size*seq_len])], vocab_size)

		self.loss = tf.reduce_sum(cross_entropy) / batch_size / seq_len
		
		tvars = tf.trainable_variables()
		opt = tf.train.AdamOptimizer(self.learning_rate)

		# calculating gradients by clipping
		grads , _  = tf.clip_by_global_norm(
				tf.gradients(self.loss, tvars), self.grad_clip)

		# applying gradients
		self.train_step = opt.apply_gradients(zip(grads, tvars))	
		
		# summaries
		tf.summary.histogram('last_state',self.final_state)
		tf.summary.histogram('logits',self.logits)
		tf.summary.histogram('predictions',self.probs)
		tf.summary.scalar('loss',self.loss)
		tf.summary.scalar('learning_rate',self.learning_rate)
		tf.summary.histogram('gradients', _)

		self.summary_op = tf.summary.merge_all()

		self.saver = tf.train.Saver()
		return(self)



