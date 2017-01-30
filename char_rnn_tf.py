import numpy as np
import tensorflow as tf

import time
import os
from six.moves import cPickle

from utils import TextLoader
from model import Model

# from model import Args

# ========= PARAMS =========

class Args():
	# RNN
	rnn_size =  10
	num_layers =  1

	# Training
	learning_rate = 0.01
	grad_clip = 5
	num_epochs = 10
	decay_rate = 0.97

	# Processing
	batch_size =  10
	seq_len =  10

	# Utils
	data_dir = 'data'
	save_dir = 'save'
	save_every = 1000
	

def main():
	args = Args()
	train(args)

def train(args):
	args = args
	data_loader = TextLoader(args.data_dir, args.batch_size, args.seq_len)
	args.vocab_size = data_loader.vocab_size

	with open(os.path.join(args.save_dir, 'config.pkl'),'wb') as f:
		cPickle.dump(args,f)
	with open(os.path.join(args.save_dir, 'chars_vocab.pkl'),'wb') as f:
		cPickle.dump((data_loader.chars, data_loader.vocab_size),f)
	model = Model()
	
	with tf.Session() as sess:
		init = tf.global_variables_initializer()
		saver = tf.train.Saver(tf.global_variables())

		for e in range(args.epochs):
			sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
			state = sess.run(model.initial_state)
			
			for b in range(args.num_batches):
				start = time.time()
				x,y =  data_loader.next_batch()
				feed = {input_data:x, targets:y}
				for i, (c,h) in enumerate(model.initial_state):
					feed[c] = state[i].c
					feed[h] = state[i].h
				summary_op, train_loss, state, _ = sess.run([model.summary_op,model.cost, model.final_state, model.train_op], feed)
				end = time.time()
				print("epoch {}, loss = {:.3f}".format(args.epochs, args.train_loss))

if __name__ == '__main__':
	main()




