#importing libraries

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import decimal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
import h5py


#functioning for preprocessing dataset A and creating X and Y numpy arrays

def preprocess(dz):
	'''
	preprocess the data of h5 file and convert into numpy array of input and output features

	input H5py file opened

	output X and Y 

	where X is array of input features 
	while Y is array of output converted into 1 D array

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
	return X,Y_l

def frequency_count(train_Y,test_Y):
	'''

	count the frequency of each class belonging to train_X and test_X and compares the ratio of sampling done

	input output features of training and test data
	i.e.

	train_Y and test_Y respectively

	output - none

	'''
	count_train=[]
	count_test=[]

	print("ratio of train to test for classes")
	for i in range(10):
	  count_train.append(np.count_nonzero(train_Y==i))
	  count_test.append(np.count_nonzero(test_Y==i))
	  print(count_train[-1]/count_test[-1])
	print("train count",*count_train)
	print("test count" ,*count_test)


def apply_PCA(train_X,test_X,variation = 0.95):

	'''

	apply PCA and normalization and transform the data

	input -training data of input features  train_X
	test data of input features test_X
	
	optional input 
	variation -  float between 0 and 1 ( set default to 0.95)
	* shows the variation for PCA data you want to have.

	output - none ( does upadtion of matrices internally (call by reference))
	'''

	#scaling and normalizing the data

	scaler = StandardScaler()
	scaler.fit(train_X)
	train_X = scaler.transform(train_X)
	test_X = scaler.transform(test_X)

	#applyng PCA with 95%variation so 147 dimensions remian out of 784

	pca = PCA(variation)
	pca.fit(train_X)
	train_X = pca.transform(train_X)
	test_X = pca.transform(test_X)

def apply_SVD(train_X,test_X,ft=100):

	#scaling and normalizing the data

	'''

	apply SVD and normalization and transform the data

	input -training data of input features  train_X
	test data of input features test_X
	
	optional input 
	ft -  integer ( set to default 100)
	* shows the final number of features for SVD data you want to have.

	output - none ( does upadtion of matrices internally (call by reference))

	'''	

	scaler = StandardScaler()
	scaler.fit(train_X)
	train_X = scaler.transform(train_X)
	test_X = scaler.transform(test_X)

	#applying SVD with parameter =100 to reduce number of rows to 100
	svd = TruncatedSVD(ft)
	svd.fit(train_X)
	train_X =svd.transform(train_X)
	test_X =svd.transform(test_X)

def apply_logistic_regression(train_X,train_Y,test_X,test_Y,itert=10000):
	'''
	applies logistic regression

	input-
	train_X - input training data
	train_Y - output training data
	test_X - input test data
	test_Y - output test data

	optional iter- no .of iterations for logistic regression ( default set to 10000)

	output - none ( prints accuracy)
	'''

	lr= LogisticRegression(max_iter=itert)
	lr.fit(train_X,train_Y)

	print("accuracy: ",lr.score(test_X,test_Y)*100)

def tSNE(train_X):
	'''
	apply TSNE and plot graph

	input train_X
	output - none just plots
	'''
	sns.set(rc={'figure.figsize':(11.7,8.27)})
	palette = sns.color_palette("bright", 10)


	t= TSNE()
	x_tsne = t.fit_transform(train_X)
	sns.scatterplot(x_tsne[:,0], x_tsne[:,1], hue=train_Y, legend='full', palette=palette)
	plt.show()
#opening file A
dz= h5py.File('part_A_train.h5','r')
X,Y = preprocess(dz)
train_X,test_X,train_Y,test_Y = train_test_split(X, Y, test_size=0.2, stratify=Y)
frequency_count(train_Y,test_Y)

print("PCA")
apply_PCA(train_X,test_X)
apply_logistic_regression(train_X,train_Y,test_X,test_Y)
tSNE(train_X)

print("SVD")
train_X,test_X,train_Y,test_Y = train_test_split(X, Y, test_size=0.2, stratify=Y)
apply_SVD(train_X,test_X)
apply_logistic_regression(train_X,train_Y,test_X,test_Y)
tSNE(train_X)