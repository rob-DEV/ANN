# imports 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_dataset():
    # Read in the .csv file and split at 0.5 train test
    data_frame = pd.read_csv('Mixcancer.csv')
    data = data_frame.to_numpy()
    
    x = data[:,1:]
    y = data[:,0]

    # Split the data ratio of 0.5 train / test
    sample_size = x.shape[0]
    train_indices = np.random.choice(range(sample_size), int(sample_size / 2), replace=False)
    selection_mask = np.zeros(sample_size, dtype=bool)
    selection_mask[train_indices] = True

    x_train = x[selection_mask]
    y_train = y[selection_mask]

    x_test = x[~selection_mask]
    y_test = y[~selection_mask]

    return x_train, y_train, x_test, y_test


class ANN:
    def __init__(self, batch_size=128, learning_rate=0.01, epochs=500):
        # Hyper parameters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs

        # For curve plotting
        self.loss_history = []

    def sigmoid(self, x):
        return 1.0/(1.0+np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x)*(1.0-self.sigmoid(x))

    def reLu(self, v):
        return np.maximum(v,0)

    def reLu_der(self, v):
        v[v<=0] = 0
        v[v>0] = 1
        return v

    def cross_entropy(self, x, y):
        return (-y*(np.log(x)) - (1-y) * np.log(1-x))

    def cross_entropy_derivative(self, x, y):
        return -(y/x - (1-y)/(1-x))

    def train(self, x_train, y_train):

        y_train = y_train.reshape(len(y_train),1)
        # Initialize weights and bias
        # self.w1 = np.random.uniform(-1.0, 1.0, (x_train.shape[1], 5))
        # self.b1 = np.zeros((1, 5))
        # self.w2 = np.random.uniform(-1.0, 1.0, (5, 1))
        # self.b2 = 0

        w1 = np.random.uniform(-1,1,[len(x_train[0]),4]) #weights of the first layer
        b1 = np.zeros([1,4]) #bias of the first layer
        w2 = np.random.uniform(-1,1,[4,1]) #weights of the first layer
        b2 = 0 #bias of the first layer
                
        train_E = []
        test_E = []
        train_Acc = []
        test_Acc = []

        for epoch in range(self.epochs):  
            #feedforward
            in1 = x_train@w1 + b1
            o1 = self.sigmoid(in1)
            in2 = o1@w2 + b2
            o2 = self.sigmoid(in2)
            
            #Evaluation
            
            #Error
            error = self.cross_entropy(o2 ,y_train).mean()
            train_E.append(error)
            # test_E.append(self.cross_entropy(self.sigmoid(self.reLu(x_test@w1+b1)@w2+b2),y_test).mean())
            # pred_test = np.where(sigmoid(reLu(x_test@w1+b1)@w2+b2) > 0.5,1,0)
            # test_Acc.append(metrics.accuracy_score(y_test,pred_test))

            #Accuracy
            pred_train = np.where(o2 > 0.5, 1,0)
            accuracy = np.sum(pred_train == y_train) / float(y_train.shape[0])
            train_Acc.append(accuracy)
            print("Accuracy: ", accuracy)

      
            
            #backpropagation Layer 2
            dE_dO2 = self.cross_entropy_derivative(o2, y_train)
            dO2_dIn2 = self.sigmoid_derivative(in2)
            dIn2_dW2 = o1
            dIn2_B2 = 1
            dE_dW2 = (1/x_train.shape[0])*dIn2_dW2.T@(dE_dO2*dO2_dIn2)
            dE_dB2 = (1/x_train.shape[0])*np.ones([1,len(x_train)])@(dE_dO2*dO2_dIn2)
            
            #backpropagation Layer 1
            dIn2_dO1 = w2
            dO1_dIn1 = self.sigmoid_derivative(in1)
            dIn1_dW1 = x_train
            dE_dW1 = (1/x_train.shape[0])*dIn1_dW1.T@((dE_dO2*dO2_dIn2@dIn2_dO1.T)*dO1_dIn1)
            dE_dB1 = (1/x_train.shape[0])*np.ones([len(x_train)])@((dE_dO2*dO2_dIn2@dIn2_dO1.T)*dO1_dIn1)
            
            #updating parameters
            b2-= self.learning_rate * dE_dB2
            w2-= self.learning_rate * dE_dW2
            b1-= self.learning_rate * dE_dB1
            w1-= self.learning_rate * dE_dW1
            
        z=np.arange(self.epochs)
        f1=plt.figure(1)
        plt.plot(z,train_E,label="train",color='red')
        # plt.plot(z,test_E,label="test",color='blue')
        plt.legend(loc='best')
        plt.title('Error')
        f1.show()

        f2=plt.figure(2)
        plt.plot(z,train_Acc,label="train",color='red')
        # plt.plot(z,test_Acc,label="test",color='blue')
        plt.legend(loc='best')
        plt.title('Accuracy')
        f2.show()

        plt.show()

def main():
    x_train, y_train, x_test, y_test = load_dataset()

    ann = ANN(batch_size=128, learning_rate=0.00000005, epochs=300)

    ann.train(x_train, y_train)

    # ann.evaluate(x_test, y_test)
    
    

main()