import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import decimal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn import tree
import h5py

def preprocess(dz):
	'''
	preprocesses the data
	'''
	df_X = pd.DataFrame(np.array(dz['X']))
	df_Y= pd.DataFrame(np.array(dz['Y']))
	X = df_X.to_numpy()
	Y = df_Y.to_numpy()
	Y_l = np.array([])
	for i in range(len(Y)):
		for j in range(len(Y[0])):
			if Y[i][j]==1:
				Y_l = np.append(Y_l,j)
				break;
	return X,Y
def train_test_split(X,Y):
	'''
	split the data into 80 -20
	'''
	train = X[:8*len(X)//10,:]
	test = X[8*len(X)//10:,:]
	train_Y = Y[:8*len(Y)//10,:]
	test_Y = Y[8*len(Y)//10:,:]
	return train,train_Y,test,test_Y

def prob_c(train_Y):
	'''calculate P(C) for training data'''
	p_c = np.array([])
	for i in range(len(train_Y[0])):
		p_c = np.append(p_c,np.sum(train_Y[:,i:i+1])/len(train_Y))
	print(p_c,p_c.sum())
	return p_c

def mean_variance(X,Y):
	'''Calculate mean and variance for the input data'''
	m_X = np.empty((len(X[0]),len(Y[0])))
	var_X = np.empty((len(X[0]),len(Y[0])))
	for i in range(len(X[0])):
		for j in range(len(Y[0])):
			z=np.array([])
			for k in range(len(X)):
				if Y[k][j]==1:
					z = np.append(z,X[k][i])
			m_X[i][j]= np.mean(z)
			var_X[i][j] = np.var(z)*len(z)/(len(z)-1)
	return m_X,var_X

def prob(m_X,var_X,i,j,xi):
	'''calculate postreior probability'''
	if var_X[i][j]==0:
		var_X[i][j]=0.0000000000001
	return (1/math.sqrt(2*math.pi*var_X[i][j]))*math.exp((-1*(xi-m_X[i][j])**2)/(2*var_X[i][j]))
def predict(X,m_x,var_X,p_c):
	'''predict the class by chosing the class with max probability'''
	pred = np.array([])
	for i in range(len(X)):
		z=np.array([])
		for j in range(len(Y[0])):
			pro_x_c =decimal.Decimal(1)
			for k in range(len(X[0])):
				pro_x_c = pro_x_c * decimal.Decimal(prob(m_X,var_X,k,j,X[i][k]))
			pro_x_c = pro_x_c*decimal.Decimal(p_c[j])
			z=np.append(z,pro_x_c)
		c_pred = np.argmax(z)
		pred = np.append(pred,c_pred)
	return pred

def accuracy(pred,Y):
	'''print accuracy between predicted and actual'''
	acc=0
	for i in range(len(pred)):
		if Y[i][int(pred[i])]==1:
			acc+=1
	print(acc/len(Y)*100)


dz= h5py.File('part_B_train.h5','r')
X,Y = preprocess(dz)
train,train_Y,test,test_Y = train_test_split(X,Y)
p_c =prob_c(train_Y)
m_X,var_X = mean_variance(train,train_Y)
p =predict(test,m_X,var_X,p_c)
accuracy(p,test_Y)