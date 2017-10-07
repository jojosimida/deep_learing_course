import theano, time
import theano.tensor as T
import numpy as np
from random import random
import matplotlib.pyplot as plt

#Define variables:
x = T.matrix('x')
w1 = theano.shared(np.array([random(),random()]))
w2 = theano.shared(np.array([random(),random()]))
w3 = theano.shared(np.array([random(),random()]))
b1 = theano.shared(0.1)
b2 = theano.shared(0.1)
learning_rate = 0.01

#Define mathematical expression:
a1 = 1/(1+T.exp(-T.dot(x,w1)-b1))
a2 = 1/(1+T.exp(-T.dot(x,w2)-b1))
x2 = T.stack([a1,a2], axis=1)
a3 = 1/(1+T.exp(-T.dot(x2,w3)-b2))

a_hat = T.vector('a_hat') #Actual output
cost = -(a_hat*T.log(a3) + (1-a_hat)*T.log(1-a3)).sum()

dw1,dw2,dw3,db1,db2 = T.grad(cost,[w1,w2,w3,b1,b2])

train = theano.function(
    inputs = [x,a_hat],
    outputs = [a3,cost, w1,w2,w3,b1,b2],
    updates = [
        [w1, w1-learning_rate*dw1],
        [w2, w2-learning_rate*dw2],
        [w3, w3-learning_rate*dw3],
        [b1, b1-learning_rate*db1],
        [b2, b2-learning_rate*db2]
    ]
)

test = theano.function(inputs = [x], outputs = [a3])


#Define inputs and weights
inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]	
outputs = [0,1,1,0]

#Iterate through all inputs and find outputs:
cost_list = []
tic = time.clock()

for iteration in range(50000):
    pred, cost_iter,weight1,weight2,weight3, bias1,bias2 = train(inputs, outputs)
    cost_list.append(cost_iter)
    
print()
print()
print('Time for training spent {} secs' .format(time.clock() - tic))
print()

#Print the outputs:
print('The outputs of the XOR gate NN are:')
for i in range(len(inputs)):
    print('The output for ({},{}) is {:.2f}'.\
    	format(inputs[i][0],inputs[i][1],pred[i]))
 

# print(weight1)
# print(weight2)
# print(weight3)
# print(bias1)
# print(bias2)

#Plot the flow of cost:
plt.figure()
plt.title('Training Curve', fontsize=20)
plt.xlim(0,len(cost_list))
plt.xlabel('Number of training')
plt.ylabel('Cost')
plt.plot(cost_list)
plt.grid(True)


# classify the and gate output
plt.figure()
plt.title('XOR gate', fontsize=20)
plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)
plt.xticks([0,1])
plt.yticks([0,1])
plt.xlabel('Input1')
plt.ylabel('Input2')

x0_list = [0,1]
y0_list = [0,1]
x1_list = [1,0]
y1_list = [0,1]
plt.scatter(x0_list, y0_list, s=75, c='B', alpha=.5, label='Off')
plt.scatter(x1_list, y1_list, s=75, c='G', alpha=.5, label='On')

# plot the linear classification of and gate
xx = np.linspace(-1,2)

# By "z = T.dot(x,w)+b" , when z=0
yy1 = (-bias1-weight1[0]*xx)/weight1[1]
yy2 = (-bias1-weight2[0]*xx)/weight2[1]
plt.plot(xx,yy1,c='R')
plt.plot(xx,yy2,c='R')

# predict testing data, the testing data is random

testing_data = [[1,1], [0,1], [0,0], [1,0]]
predictions = test(testing_data)
print()
print("The testing data are: {}".format(testing_data))
print("The predictions of the testing data: \n{}".format(predictions[0]))


plt.legend()
plt.show()