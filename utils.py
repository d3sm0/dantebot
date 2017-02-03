import numpy as np

def load_data(file='data/dante.txt'):
	with open(file, 'r') as f:
	    raw_data = f.read()
	    print('Data loaded. Size of the dataset is {}'.format(len(raw_data)))
	return raw_data

def parse_data(raw_data):
	vocab = set(raw_data)
	vocab_size = len(vocab)
	idx_to_vocab = dict(enumerate(vocab))
	vocab_to_idx = dict(zip(idx_to_vocab.values(), idx_to_vocab.keys()))
	data = np.array([vocab_to_idx[c] for c in raw_data])
	return (vocab_size, data, (idx_to_vocab, vocab_to_idx))

def create_batches(data, seq_len = 20, batch_size = 50):
	num_batches = int(len(data)/seq_len/batch_size)
	new_data = data[:num_batches*batch_size*seq_len]
	xdata = new_data
	ydata = np.copy(xdata)

	# building up the sequence
	ydata[:-1] = xdata[1:]
	ydata[-1] = xdata[0]

	xdata = xdata.reshape(batch_size, -1)
	ydata = ydata.reshape(batch_size, -1)

	# create batches
	x_batches = np.split(xdata, num_batches,1)
	y_batches = np.split(ydata, num_batches,1)

	return(x_batches, y_batches, batch_size)

def enc_dec(vocab):
	
	return(idx_to_vocab, vocab_to_idx)
