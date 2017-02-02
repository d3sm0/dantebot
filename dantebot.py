import tensorflow as tf

from model import Model
import utils

import time


class Args():
	batch_size = 50
	seq_len = 25
	rnn_size = 20
	learning_rate = 1e-2
	global_dropout = 0.98
	num_layers = 2
	num_epochs = 5

def main():
	args = Args()
	vocab_size, data = utils.parse_data(utils.load_data())
	x_, y_, num_batches = utils.create_batches(data, args.seq_len, args.batch_size)
	args.vocab_size = vocab_size
	args.num_batches = num_batches

	train_network(x_, y_,args)


def train_network(x_,y_,args):
	
	model = Model(args)
	graph = model.build_graph(args)

	with tf.Session() as sess:
		
		sess.run(tf.global_variables_initializer())
		train_writer = tf.summary.FileWriter('train',sess.graph)

		steps = 0
		tr_losses = []
		training_state = None
		
		for e in range(1,args.num_epochs):

			tr_loss = 0 
			for p in range(args.num_batches):
				start = time.time()
				feed = {graph['x'] : x_[p], graph['y'] : y_[p]}

				if training_state is not None:
					feed[graph['init']] = training_state

				loss_, training_state, _ = sess.run([graph['loss'], 
					graph['final_state'],  graph['train_step']],feed)
				
				tr_loss += loss_
				end = time.time()

				if p % 10 == 0:

					tr_losses.append(tr_loss)
					summary_op = sess.run(graph['summary_op'], feed)
					train_writer.add_summary(summary_op, e*data_loader.num_batches+b)

					print('Average loss {}, per batch {} at epoch {}, time per batch {}'.format(tr_loss, p, e, end-start))

			graph['saver'].save(sess, 'save/', global_step = e*args.num_batches+p)


if __name__ == '__main__':
	main()
