import theano, time
import theano.tensor as T
import numpy as np
from random import random
import matplotlib.pyplot as plt


MIN_LENGTH = 50
MAX_LENGTH = 55

x_seq = T.matrix()
a_0 = T.matrix()
# learning rate
lr = T.scalar()
Wi = T.matrix()
Wh = T.matrix()
Wo = T.matrix()
bh = theano.shared(0.1)
bo = theano.shared(0.1)
y_hat_seq = T.iscalar()


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

    return x_seq, y_hat, length


# recurrent function (using tanh activation function) and linear output
# activation function
def step(x_t, a_tm1, Wi, Wh, Wo):
    a_t = T.tanh(T.dot(x_t, Wi) + T.dot(a_tm1, Wh) + bh)
    y_t = T.nnet.softmax(T.dot(a_t, Wo) + bo)

    return a_t, y_t


[a_seq, y_seq], _ = theano.scan(step,
                        sequences=x_seq,
                        outputs_info=[a_0, None],
                        non_sequences=[Wi, Wh, Wo])
# error between output and target
error = ((y_seq - y_hat_seq) ** 2).sum()
# gradients on the weights using BPTT
gWi, gWh, gWo, gbh, gbo= T.grad(error, [Wi, Wh, Wo, bh, bo])
# training function, that computes the error and updates the weights using
# SGD.
rnn = theano.function([a_0, x_seq, y_hat_seq, lr],
                     error,
                     updates={Wi: Wi - lr * gWi,
                             Wh: Wh - lr * gWh,
                             Wo: Wo - lr * gWo,
                             bh: bh - lr * gbh,
                             bo: bo - lr * gbo
                             })


for i in range(2):
    x_seq, y_hat_seq, length = gen_data()
    length = 50
    Wi = theano.shared(np.random.uniform(size=(length, 2), low=-.01, high=.01))
    Wh = theano.shared(np.random.uniform(size=(length, 1), low=-.01, high=.01))
    Wo = theano.shared(np.random.uniform(size=(length, 1), low=-.01, high=.01))
    a_0 = theano.shared(np.random.uniform(size=(length, 1), low=-.01, high=.01))

    print(rnn(a_0,x_seq,y_hat_seq,0.01))

    # print(Wi.get_value().shape)
    # print(Wi.get_value())

