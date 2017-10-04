import theano, time
import theano.tensor as T
import numpy as np
from random import random
import matplotlib.pyplot as plt

#Define variables:
x = T.matrix('x')
w = theano.shared(np.array([random(),random()]))
b = theano.shared(1.)
learning_rate = 0.01

#Define mathematical expression:
z = T.dot(x,w)+b
a = 1/(1+T.exp(-z))

a_hat = T.vector('a_hat') #Actual output
cost = -(a_hat*T.log(a) + (1-a_hat)*T.log(1-a)).sum()

dw,db = T.grad(cost,[w,b])

neuron = theano.function(
    inputs = [x,a_hat],
    outputs = [a,cost,w,b],
    updates = [
        [w, w-learning_rate*dw],
        [b, b-learning_rate*db]
    ]
)


#Define inputs and weights
inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]	
outputs = [0,0,0,1]

#Iterate through all inputs and find outputs:
cost_list = []
tic = time.clock()

for iteration in range(50000):
    pred, cost_iter,weight, bias = neuron(inputs, outputs)
    cost_list.append(cost_iter)
    

print('time spent evaluating one value {} sec' .format(time.clock() - tic))
print()

#Print the outputs:
print('The outputs of the NN are:')
for i in range(len(inputs)):
    print('The output for x1={} | x2={} is {:.2f}'.\
    	format(inputs[i][0],inputs[i][1],pred[i]))
   
#Plot the flow of cost:
print('\nThe flow of cost during model run is as following:')

plt.plot(cost_list)

# classify the and gate output
plt.figure()
plt.title('AND gate', fontsize=20)
plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)
plt.xticks([0,1])
plt.yticks([0,1])
plt.xlabel('Input1')
plt.ylabel('Input2')

x0_list = [0,0,1]
y0_list = [0,1,0]
plt.scatter(x0_list, y0_list, s=75, c='B', alpha=.5, label='Off')
plt.scatter(1, 1, s=75, c='G', alpha=.5, label='On')

# plot the linear classification of and gate
xx = np.linspace(0,2)

# By "z = T.dot(x,w)+b" , when z=0
yy = (-bias-weight[0]*xx)/weight[1]
plt.plot(xx,yy,c='R')

# predict testing data, the testing data is random
rand = [random(),random()]
T = 'G' if np.dot(rand,weight)+bias>=0 else 'B'
plt.scatter(rand[0],rand[1], s=75, c=T, alpha=.5)
plt.annotate('predict testing data', xy=rand, 
             xytext=(+30, -30), textcoords='offset points', fontsize=16,
             arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2"))

plt.legend()
plt.show()