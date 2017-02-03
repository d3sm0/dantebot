import utils
import tensorflow as tf
import numpy as np

def talk(graph, enc, num_chars, save_dir='save/', prompt='e', a=5):

	id_vocab = enc[0]
	vocab_id = enc[1] 

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		# loading ckpt dir from save_dir
		ckpt = tf.train.get_checkpoint_state('save/')

		# load model
		graph['saver'].restore(sess, ckpt.model_checkpoint_path)

		state = None
		char_id = vocab_id[prompt]
		chars = [char_id]

		for ch in range(num_chars):

			if state != None:
				feed = {graph['x']:[[char_id]], graph['init']:state}
			else:
				feed = {graph['x']:[[char_id]]}
			preds, state = sess.run([graph['predictions'], graph['final_state']], feed)

			# return (81,) array
			p = np.squeeze(preds)

			# last 5 probs are the best
			p_sort = np.argsort(p)

			# lets' remove all exepct the top 5
			p[p_sort[:-a]]=0

			# adjust probs
			p = p /np.sum(p)

			# return id of chosen char
			char_id = np.random.choice(len(vocab_id),1, p=p)[0]

			chars.append(char_id)

	chars = map(lambda x: id_vocab[x], chars)

	return(''.join(chars))

