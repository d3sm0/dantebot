import utils
import tensorflow as tf
import numpy as np

def talk(g, chars, vocab, num=200, prime='La ', save_dir='save/'):

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver(tf.global_variables())
		ckpt = tf.train.get_checkpoint_state(save_dir)
		saver.restore(sess, ckpt.model_checkpoint_path)

		# init sampling

		state = sess.run(g.cell.zero_state(1, tf.float32))

		for char in prime[:-1]:
			x = np.zeros((1,1))
			x[0,0] = vocab[char]
			feed = {g.x: x, g.init :state}
			state = sess.run([g.final_state], feed)

		def weighted_pick(weights):
			t = np.cumsum(weights)
			s = np.sum(weights)
			
			return(int(np.searchsorted(t, np.random.rand(1)*s)))

		ret = prime
		# blank space
		char = prime[-1]
		for n in range(num):
			x = np.zeros((1,1))
			x[0,0] = vocab[char]

			feed = {g.x:x, g.init:state}
			probs, state = sess.run([g.probs, g.final_state], feed)

			p = probs[0]

			sample = weighted_pick(p)

			pred = chars[sample]
			ret += pred
			char = pred
		return ret