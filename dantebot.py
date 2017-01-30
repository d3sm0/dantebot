import numpy as np
import tensorflow as tf

import time
import os
from six.moves import cPickle

from utils import TextLoader
from model import Model

# from model import Args

class Args():
	# RNN
	rnn_size =  128
	num_layers =  3

	# Training
	learning_rate = 0.01
	grad_clip = 5.
	num_epochs = 50

	# Processing
	batch_size =  50
	seq_len =  50

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
		cPickle.dump((data_loader.chars, data_loader.vocab),f)

	model = Model(args)
	
	with tf.Session() as sess:
		
		sess.run(tf.global_variables_initializer())

		train_writer = tf.summary.FileWriter('train',sess.graph)
		saver = tf.train.Saver(tf.global_variables())

		for e in range(args.num_epochs):

			sess.run(tf.assign(model.lr, args.learning_rate))

			data_loader.reset_batch_pointer()
			
			state = sess.run(model.init_state)
			for b in range(data_loader.num_batches):
				start = time.time()
				x,y =  data_loader.next_batch()
				feed = {model.x: x, model.y: y}

				for i, (c,h) in enumerate(model.init_state):
					feed[c] = state[i].c
					feed[h] = state[i].h

				train_loss, state, _ = sess.run([model.cost, model.final_state, model.train_op], feed)
				if(e*data_loader.num_batches+b) % args.save_every == 0:
					summary_op = sess.run(model.summary_op, feed)
					train_writer.add_summary(summary_op, e*data_loader.num_batches+b)
					end = time.time()
					saver.save(sess, checkpoint_path, global_step = e*data_loader.num_batches + b)
				print("{}/{}, epoch {} train_loss = {:.3f}, time/batch = {:.3f}".format(
					e*data_loader.num_batches + b,args.num_epochs*data_loader.num_batches, 
					e ,args.num_epochs, train_loss, end-start))

if __name__ == '__main__':
	main()




