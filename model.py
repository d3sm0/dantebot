import tensorflow as tf

from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import seq2seq

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

class Model():
	def __init__(self, args, infer = False):
		self.args = args

		cell = rnn_cell.LSTMCell(args.rnn_size, state_is_tuple = True)
		self.cell = cell = rnn_cell.MultiRNNCell([cell] * args.num_layers, state_is_tuple = True)

		with tf.name_scope('input'):
			self.x = tf.placeholder(tf.int32, [args.batch_size, args.seq_len])
			self.y = tf.placeholder(tf.int32, [args.batch_size, args.seq_len])

	# init state of the cells as 0
		self.init_state = cell.zero_state(args.batch_size, tf.float32)

		with tf.variable_scope('rnn'):
				softmax_w = tf.get_variable('softmax_w', [args.rnn_size, args.vocab_size])
				softmax_b = tf.get_variable('softmax_b', [args.vocab_size])

		# embedding
		with tf.device('/cpu:0'):
			embedding = tf.get_variable('embedding', [args.vocab_size, args.rnn_size])
			input_lookup = tf.nn.embedding_lookup(embedding, self.x)
			inputs = tf.split(1, args.seq_len, input_lookup)
			# inputs list
			inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

		def loop(prev, _):
		    prev = tf.matmul(prev, softmax_w) + softmax_b
		    prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
		    return tf.nn.embedding_lookup(embedding, prev_symbol)

		with tf.name_scope('decoder'):
			outputs, last_state = seq2seq.rnn_decoder(inputs, 
			    self.init_state, cell, scope = 'rnn')

		with tf.name_scope('Wx_plus_b'):
			self.output = tf.reshape(tf.concat(1, outputs),[-1,args.rnn_size])
			self.logits = tf.matmul(self.output, softmax_w) + softmax_b
			tf.summary.histogram('pre_activations', self.logits)

		with tf.name_scope('softmax'):
			self.probs = tf.nn.elu(self.logits)
			tf.summary.histogram('activations', self.probs)

		with tf.name_scope('cross_entropy'):
			cross_entropy = seq2seq.sequence_loss_by_example([self.logits],
				[tf.reshape(self.y,[-1])],
				[tf.ones([args.batch_size * args.seq_len])],
				args.vocab_size)
			tf.summary.histogram('cross_entropy', cross_entropy)

		with tf.name_scope('train_loss'):
			# not sure that this make sense
			self.cost = tf.reduce_sum(cross_entropy) / args.batch_size / args.seq_len
			tf.summary.scalar('train_loss',self.cost)

		with tf.name_scope('state'):
			self.final_state = last_state
			tf.summary.histogram('last_state',self.final_state)

		with tf.name_scope('train'):
			
			self.lr = tf.Variable(0.0, trainable=False)
			tf.summary.scalar('lr',self.lr)
			
			tvars = tf.trainable_variables()
			grads , _ = tf.clip_by_global_norm(
				tf.gradients(self.cost, tvars), args.grad_clip)

			opt = tf.train.AdamOptimizer()
			# backprop
			self.train_op = opt.apply_gradients(zip(grads,tvars))

		self.summary_op = tf.summary.merge_all()
