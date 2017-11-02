import theano, time
import theano.tensor as T
import numpy as np
from random import random
import matplotlib.pyplot as plt

MIN_LENGTH = 50
MAX_LENGTH = 55
N_input = 2    
N_hidden = 2
N_output = 2  

bh = theano.shared(np.zeros(N_hidden))
bo = theano.shared(np.zeros(N_output))
Wi = theano.shared(np.random.uniform(size=(N_input,N_hidden), low=-.01, high=.01))
Wh = theano.shared(np.random.uniform(size=(N_hidden,N_hidden), low=-.01, high=.01))
Wo = theano.shared(np.random.uniform(size=(N_hidden,N_output), low=-.01, high=.01))
a_0 = theano.shared(np.zeros(N_hidden))
# y_0 = theano.shared(np.zeros(N_output))
x_seq = T.matrix()
y_hat_seq = T.scalar()
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



def step(x_t, a_tm1):
	a_t = T.tanh(T.dot(x_t,Wi) + T.dot(a_tm1,Wh) + bh)
	# y_t_ori = T.nnet.softmax(T.dot(a_t,Wo) + bo)
	# y_t = T.argmax(y_t_ori)
	y_t =  T.nnet.softmax(T.dot(a_t,Wo) + bo)


	return a_t,y_t


[a_seq,y_seq], _ = theano.scan(
					step,
					sequences = x_seq,
					outputs_info = [a_0,None],
					truncate_gradient=-1
					
					)


# y_seq_last = y_seq[-1]

cost = T.sum((y_seq - y_hat_seq)**2)
gWi, gWh, gWo, gbh, gbo = T.grad(cost, [Wi,Wh,Wo,bh,bo])


rnn_train = theano.function(
			inputs=[x_seq,y_hat_seq],
			outputs=[cost, y_seq],
			updates = [
					[Wi, Wi-learning_rate*gWi],
					[Wh, Wh-learning_rate*gWh],
					[Wo, Wo-learning_rate*gWo],
					[bh, bh-learning_rate*gbh],
					[bo, bo-learning_rate*gbo],
			],
			)

rnn_test = theano.function(inputs= [x_seq],outputs=y_seq)



epochs = 10000

for i in range(epochs):
 	x_seq, y_hat_seq = gen_data()

 	c, y= rnn_train(x_seq, y_hat_seq)

 	if i%1000==0:
 		print("iteration: {} ,cost: {}".format(i,c))
 		print(y)
 		# print(wi)
 		# print(wh)
 		# print(wo)
 		print()


# for i in range(10):
# 	x_seq, y_hat = gen_data()
# 	aa = rnn_test(x_seq)
# 	print("Answer: {}, prediction: {}".format(y_hat, aa))
