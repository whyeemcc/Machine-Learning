import sys,os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = sys.path[0]
csv_name = 'watermelon sample.csv'
data = pd.read_csv(path + '/' + csv_name, header=0, encoding='utf-8')

# sample count
m = len(data)

X = np.array([data['density'],
              data['sugar']]
            )
Y = np.array(data['result'])

# sigmoid function
def sigmoid(z):
    return 1/(1 + np.exp(-z))

# loss function
def loss(a,y):
    return -(y*np.log(a)+(1-y)*np.log(1-a))
    
# initialize the w and b
w = np.zeros((2,1))
b = 0

# learning rate
alpha = 2

loop = 1000
Loss_list = []
for i in range(loop):
    Z = np.dot(w.T,X) + b
    A = sigmoid(Z)
    # loss value
    L = 1/m*np.sum(loss(A,Y))
    Loss_list.append(L)
    # update parameters
    dZ = A - Y
    dw = 1/m*np.dot(X,dZ.T)
    db = 1/m*np.sum(dZ)
    w = w - alpha*dw
    b = b - alpha*db

red = data[data.result == 1]
blue = data[data.result == 0]    
# plot Loss    
plt.figure()  
ax1 = plt.subplot(211)        
ax1.plot(Loss_list)
ax1.set_title('Loss function')
# plot result
ax2 = plt.subplot(212)
x = np.arange(0.1,0.9,0.01)
y = -(w[0]*x+b)/w[1]
plt.plot(x,y,c='orange')
plt.scatter(red['density'],red['sugar'],c='red',label='good')
plt.scatter(blue['density'],blue['sugar'],marker='x',c='blue',label='bad')
plt.xlabel('density')     
plt.ylabel('sugar')     
plt.xlim(0.1,0.9)
plt.ylim(0,0.5)
plt.legend()
plt.show()