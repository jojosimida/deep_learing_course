import theano, time
import theano.tensor as T
import numpy as np
from random import random
import matplotlib.pyplot as plt

MIN_LENGTH = 50
MAX_LENGTH = 55

x_seq = T.matrix()
a_0 = theano.shared(random())
y_0 = theano.shared(random())
Wi = theano.shared(np.array([random(),random()]))
Wh = theano.shared(random())
Wo = theano.shared(random())
bh = theano.shared(.1)
bo = theano.shared(.1)
y_hat_seq = T.iscalar()

parameters = [Wi,Wh,Wo,bh,bo]
learning_rate = 0.01


def gen_data(min_length=MIN_LENGTH, max_length=MAX_LENGTH):
	
	# Generate x_seq
	length = np.random.randint(min_length, max_length)
	x_seq = np.concatenate([np.random.uniform(size=(length,1)),
							np.zeros((length,1))], axis=1) #equal axis=-1

	# Set the second dimension to 1 at the indices to add
	x_seq[np.random.randint(length/10),1] = 1
	x_seq[np.random.randint(length/2,length),1] = 1

	# Multiply and sum the dimension of x_seq to get the target value
	y_hat = np.sum(x_seq[:,0]*x_seq[:,1])

	return x_seq, y_hat


def step(x_t, a_tm1,y_tm1):
	a_t = T.tanh(T.dot(x_t,Wi)\
					+ T.dot(a_tm1,Wh) + bh)

	# y_t = T.nnet.softmax(T.dot(a_t,Wo) + bo)
	# y_t = T.nnet.softmax( [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0])
	y_t = T.dot(a_t,Wo) + bo

	return a_t,y_t


[a_seq,y_seq], _ = theano.scan(
					step,
					sequences = x_seq,
					outputs_info = [a_0,y_0]
				
					)

cost = T.sum((y_seq - y_hat_seq)**2)
gWi, gWh, gWo, gbh, gbo = T.grad(cost, parameters)

rnn_train = theano.function(
			inputs=[x_seq,y_hat_seq],
			outputs=cost,
			updates = [
					[Wi, Wi-learning_rate*gWi],
					[Wh, Wh-learning_rate*gWh],
					[Wo, Wo-learning_rate*gWo],
					[bh, bh-learning_rate*gbh],
					[bo, bo-learning_rate*gbo],
			],
			allow_input_downcast=True
			)


for i in range(100):
	x_seq, y_hat_seq = gen_data()
	print(rnn_train(x_seq,y_hat_seq))
	
