import torch
import torchvision
from torchvision import transforms
import torch.nn as NN
import torch.nn.functional as funct
import matplotlib.pyplot as plt
import torch.optim as optim
import pandas as pd
import numpy as np
import pickle

#Nueral network class
class Nueral_Net(NN.Module):
	#hidden units- a parameter which denotes no.of nuerons on hidden layer 
    def __init__(self,hidden_unit):
        super().__init__()
        #declaring layer 1
        self.layer_1 = NN.Linear(128, hidden_unit)
        #initializing wrights and bias of layer 1 to be random
        torch.nn.init.uniform_(self.layer_1.weight, a=0.0, b=1.0)
        torch.nn.init.uniform_(self.layer_1.bias, a=0.0, b=1.0)
        #declaring layer 3- as output layer
        self.layer_3 = NN.Linear(hidden_unit, 10)
        #initializing wrights and bias of layer 3 to be random
        torch.nn.init.uniform_(self.layer_3.weight, a=0.0, b=1.0)
        torch.nn.init.uniform_(self.layer_3.bias, a=0.0, b=1.0)
    def forward(self, x):
    	#over-riding forward function
        x = funct.relu(self.layer_1(x))
        x = self.layer_3(x)
        return x

#reading train and test files
train = pd.read_csv("largeTrain.csv",header=None)
test = pd.read_csv("largeValidation.csv",header=None)
#splitting the files to create train set and labels and test set and labe;s
train_X = torch.from_numpy(train[train.columns[1:]].to_numpy().astype('float32'))
train_Y = torch.from_numpy(train[train.columns[:1]].to_numpy())
test_X = torch.from_numpy(test[test.columns[1:]].to_numpy().astype('float32'))
test_Y = torch.from_numpy(test[test.columns[:1]].to_numpy())

#declaring arrays for learning rate and hidden units
hidden_units = [5,20,50,100,200]
learning_rates = [0.1,0.01,0.001]

#FOR hidden units
g_train = []
g_test = []
for h_U in hidden_units:
	#declaring object of the NN for pytorch
	nn_torch = Nueral_Net(h_U)
	print(nn_torch)
	#declaring loss function
	loss_function = NN.CrossEntropyLoss()
	#declaring optimzer fro backward propgation with learning rate 0.01
	optimizer = optim.Adam(nn_torch.parameters(), lr=0.01)
	#number of iterations
	epocs =100
	train_loss = []
	val_loss =[]
	for i in range(epocs): 
		X = train_X
		y= train_Y 
		#making all gradient zero
		nn_torch.zero_grad() 
		#getting output from forward propogation of first layer
		output = nn_torch(X)
		#output for test data
		val_output = nn_torch(test_X)
		#calculating losses
		loss = loss_function(output, y.T[0]) 
		loss_val = loss_function(val_output,test_Y.T[0])
		#backwad propogation
		loss.backward() 
		#gradient changer and calculater
		optimizer.step()
		train_loss.append(loss)
		val_loss.append(loss_val)
	g_train.append(train_loss[-1])
	g_test.append(val_loss[-1])
#plotting graphs
plt.plot(hidden_units,g_train,label = 'training loss')
plt.plot(hidden_units,g_test, label = 'testing loss')
plt.xlabel('no .of hidden units')
plt.ylabel('entropy loss')
plt.legend()
plt.title('HIDDEN UNITS VS ENTROPY')
plt.show()

#same things for different learning rate and hidden nodes =4
g_train = []
g_test = []
for lrtes in learning_rates:
	nn_torch = Nueral_Net(4)
	print(nn_torch)
	loss_function = NN.CrossEntropyLoss()
	optimizer = optim.Adam(nn_torch.parameters(), lr=lrtes)
	epocs =100
	train_loss = []
	val_loss =[]
	for i in range(epocs): 
		X = train_X
		y= train_Y 
		nn_torch.zero_grad() 
		output = nn_torch(X)
		val_output = nn_torch(test_X)
		loss = loss_function(output, y.T[0]) 
		loss_val = loss_function(val_output,test_Y.T[0])
		loss.backward() 
		optimizer.step()
		train_loss.append(loss)
		val_loss.append(loss_val)
	plt.plot(train_loss,label = 'training loss')
	plt.plot(val_loss, label = 'testing loss')
	plt.xlabel('epocs')
	plt.ylabel('entropy loss')
	plt.legend()
	plt.title('learning rate '+ str(lrtes)+' VS ENTROPY')
	plt.show()