## Dantebot
Dantebot is a generator Dante Alighieri sentences built with Tensorflow. 

The model is based on the char-rnn model of Andrej Karpathy. [link](http://karpathy.github.io/2015/05/21/rnn-effectiveness/). 

It uses the Divina Commedia as a data source (512 Kb txt in /data) to learn how to write in Dante's style from scratch.

## Train
To train the model
```
python dantebot.py --train
```

For visualize during training
```
tensorboard --logidir:save/
```

## Generate
To generate from the model
```
python dantebot.py --no-train
```

Example:

La quinte voglia
l'Utar mirando moidenosi da ira,
così omo!», nel fer dov' emmo uscure.
Né poi, tal girando lor nel man diverse:
con curice che feata mi fonana
prima comporta per una spetta storpa

## Model
The dynamic recurrent recurrent multi-layer neural network made of LSTM cells, with elu activation function, wrapped in a dropout layer.

(alt text)[model]

[model]: https://gitlab.com/d3sm0/dantebot/ "dantebot_model"

## Parameters

### Training
Training parameters are the following and can be manually changed in dantebot.py:

seq_len = 50
batch_size = 50
num_epochs = 20

Training data from past experience shows that over 25 epochs shows that the min loss is around 1.4.

### Model
Model parameters are the following and can be changed in the model.py:

rnn_size = 158 
num_layers = 3
learning_rate = 0.001
global_dropout = 0.9
grad_clip = 5.

Computing data from past experience shows that the model is computed in 2.14 seconds. For more information about LSTM in tensorflow check [r2rt.com](http://r2rt.com/recurrent-neural-networks-in-tensorflow-ii.html)

## Notes
The model is not fully commented, but feel free to reach out or raise an issue on the repo and I'll get back to the passage.



