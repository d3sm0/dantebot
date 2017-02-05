import tensorflow as tf

import time

def train_network(graph, data_loader, num_epochs, batch_size, seq_len, save_dir= 'save/'):
	
	tr_loss = []
	train_start = time.time()
	with tf.Session() as sess:

		sess.run(tf.global_variables_initializer())
		train_writer = tf.summary.FileWriter('save/',sess.graph)

		for e in range(num_epochs):

			data_loader.reset_batch_pointer()

			state = sess.run(graph.init)

			for b in range(data_loader.num_batches):
				start = time.time()
				x,y = data_loader.next_batch()
				feed = {graph.x: x, graph.y:y}

				for i, (c,h) in enumerate(graph.init):
					feed[c] = state[i].c
					feed[h] = state[i].h

				train_loss, final_state, _ = sess.run([graph.loss, graph.final_state,graph.train_step],feed)
				end = time.time()
				if b % 100 == 0:
					print('Latest loss {tr_loss:.{p}f}, at epoch {e}, time per batch {t:.{p}f} '.format(tr_loss = train_loss, e=e, t=end-start, p=3))

			summary_op = sess.run(graph.summary_op,feed)
			train_writer.add_summary(summary_op, e*data_loader.num_batches)

		graph.saver.save(sess, save_dir+'lstm_dec.ckpt', global_step = e*data_loader.num_batches)

	train_end = time.time() - train_start
	print('Total time {tt:.{p}}'.format(tt=train_end,p=3))
	print('For a detailed report check tensorboard --logidr=save/')