import numpy as np
from sklearn import preprocessing
from sklearn import datasets
import matplotlib.pyplot as plt

x, y = datasets.make_moons(500, noise=0.1)

from sklearn.model_selection import train_test_split

y=y.reshape(len(y),1)
#spliting data into train/test sets
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2, random_state=0, shuffle=True)

def sigmoid(v):
    return 1/(1+np.exp(-v))

def sigmoid_der(v):
    return sigmoid(v)*(1-sigmoid(v))

def reLu(v):
    return np.maximum(v,0)

def reLu_der(v):
    v[v<=0] = 0
    v[v>0] = 1
    return v

def crossEntrop(o,y):
    return (-y*(np.log(o)) - (1-y)* np.log(1-o))

def crossEntrDeriv(o,y):    
    return -(y/o - (1-y)/(1-o))

np.random.seed(42)
w1 = np.random.uniform(-1,1,[len(x[0]),4]) #weights of the first layer
b1 = np.zeros([1,4]) #bias of the first layer
w2 = np.random.uniform(-1,1,[4,1]) #weights of the first layer
b2 = 0 #bias of the first layer

from sklearn import metrics

np.random.seed(42)
w1 = np.random.uniform(-1,1,[len(x[0]),4]) #weights of the first layer
b1 = np.zeros([1,4]) #bias of the first layer
w2 = np.random.uniform(-1,1,[4,1]) #weights of the first layer
b2 = 0 #bias of the first layer

l = 0.05
epochs = 1000

train_E = []
test_E = []
train_Acc = []
test_Acc = []


for epoch in range(epochs):  
    #feedforward
    in1 = x_train@w1 + b1
    o1 = reLu(in1)
    in2 = o1@w2 + b2
    o2 = sigmoid(in2)
    
    #Evaluation
       
    #Error
    error = crossEntrop(o2 ,y_train).mean()
    train_E.append(error)
    test_E.append(crossEntrop(sigmoid(reLu(x_test@w1+b1)@w2+b2),y_test).mean())
    
    #Accuracy
    pred_train = np.where(o2 > 0.5, 1,0)
    pred_test = np.where(sigmoid(reLu(x_test@w1+b1)@w2+b2) > 0.5,1,0)
    train_Acc.append(metrics.accuracy_score(y_train,pred_train))
    test_Acc.append(metrics.accuracy_score(y_test,pred_test))
    
    #backpropagation Layer 2
    dE_dO2 = crossEntrDeriv(o2, y_train)
    dO2_dIn2 = sigmoid_der(in2)
    dIn2_dW2 = o1
    dIn2_B2 = 1
    dE_dW2 = (1/x_train.shape[0])*dIn2_dW2.T@(dE_dO2*dO2_dIn2)
    dE_dB2 = (1/x_train.shape[0])*np.ones([1,len(x_train)])@(dE_dO2*dO2_dIn2)
    
    #backpropagation Layer 1
    dIn2_dO1 = w2
    dO1_dIn1 = reLu_der(in1)
    dIn1_dW1 = x_train
    dE_dW1 = (1/x_train.shape[0])*dIn1_dW1.T@((dE_dO2*dO2_dIn2@dIn2_dO1.T)*dO1_dIn1)
    dE_dB1 = (1/x_train.shape[0])*np.ones([len(x_train)])@((dE_dO2*dO2_dIn2@dIn2_dO1.T)*dO1_dIn1)
    
    #updating parameters
    b2-=l*dE_dB2
    w2-=l*dE_dW2
    b1-=l*dE_dB1
    w1-=l*dE_dW1
    
z=np.arange(epochs)
f1=plt.figure(1)
plt.plot(z,train_E,label="train",color='red')
plt.plot(z,test_E,label="test",color='blue')
plt.legend(loc='best')
plt.title('Error')
f1.show()

f2=plt.figure(2)
plt.plot(z,train_Acc,label="train",color='red')
plt.plot(z,test_Acc,label="test",color='blue')
plt.legend(loc='best')
plt.title('Accuracy')
f2.show()


plt.show()