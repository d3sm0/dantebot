import tensorflow as tf

from model import Model
from sample import talk
import utils
import time

def main():
	
	
	train = True
	raw_data = utils.load_data()
	vocab_size, data, enc = utils.parse_data(raw_data)

	graph = Model()
		
	if train == True:

		num_epochs = 75
		seq_len = 50
		batch_size = 50

		start = time.time()
		g = graph.build_graph(batch_size, seq_len, vocab_size)
		print('It took {n:.{p}f} to build the graph. Starting training now'.format(n = time.time()-start, p=3))
		train_network(g, data, num_epochs, batch_size, seq_len)
		
	else:
		batch_size = 1
		seq_len = 1

		g = graph.build_graph(batch_size, seq_len, vocab_size)
		text = talk(g, enc , num_chars = 50)

		print('Dante says: /n' + text)

def train_network(graph, data, num_epochs, batch_size, seq_len):
	train_start = time.time()

	x_, y_, num_batches = utils.create_batches(data, seq_len, batch_size)

	with tf.Session() as sess:
		
		sess.run(tf.global_variables_initializer())
		train_writer = tf.summary.FileWriter('save/',sess.graph)

		steps = 0
		tr_losses = []
		training_state = None
		
		for e in range(1,num_epochs):

			tr_loss = 0 
			for p in range(num_batches):
				start = time.time()
				feed = {graph['x'] : x_[p], graph['y'] : y_[p]}

				if training_state is not None:
					feed[graph['init']] = training_state

				loss_, training_state, _ = sess.run([graph['loss'], 
					graph['final_state'],  graph['train_step']],feed)
				
				tr_loss += loss_
				end = time.time()

				if p % 100 == 0:

					tr_losses.append(tr_loss)
					summary_op = sess.run(graph['summary_op'], feed)
					print('Average loss {tr_loss:.{p}f}, at epoch {e}, time per batch {t:.{p}f} '.format(tr_loss = tr_loss, e=e, t=end-start, p=3))

			train_writer.add_summary(summary_op, e*num_batches)

		graph['saver'].save(sess, 'save/model_lstm.ckpt', global_step = e*num_batches)
	train_end = time.time()
	print('Train completed with av. loss {av_loss:.{p}f}, total time {t:.{p}f}, avg. time per epoch{t_e:.{p}f}'.format(av_loss = tr_losses/num_epochs, t= train_end - train_start, t_e = (train_end-train_start)/num_epochs))
	print('For a detailed report check tensorboard --logidr=save/')


if __name__ == '__main__':
	main()
