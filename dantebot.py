import tensorflow as tf

from model import Model
from sample import talk
from train import train_network

from utils import TextLoader

import time

def main(train):
	
	graph = Model()
	#vocab_size, data, enc = utils.parse_data(utils.load_data())

	seq_len = 50
	batch_size = 50

	data_loader = TextLoader('data/', batch_size, seq_len)
	vacab_size = data_loader.vocab_size
		
	if train:

		# Training default params
		num_epochs = 50

		# build graph
		start = time.time()
		g = graph.build_graph(batch_size, seq_len, vacab_size)
		print('Graph built in {n:.{p}f}. Training the network...'.format(n = time.time()-start, p=3))
		# start training
		train_network(g, data_loader, num_epochs, batch_size, seq_len)
		
	else:
		# sampling params:

		batch_size = 1
		seq_len = 1
		g = graph.build_graph(batch_size, seq_len, data_loader.vocab_size)
		print(talk(g, data_loader.chars,  data_loader.vocab))


if __name__ == '__main__':
	main(train=True)
