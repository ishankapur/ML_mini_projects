from sklearn.datasets import fetch_openml
import numpy as np
import pandas as pd
import math
import pickle
from scipy.special import softmax
from sklearn.model_selection import train_test_split



class MyNeuralNetwork():
    """
    My implementation of a Neural Network Classifier.
    """
    

    def __init__(self, n_layers, layer_sizes, activation, learning_rate, weight_init, batch_size, num_epochs):
        """
        Initializing a new MyNeuralNetwork object

        Parameters
        ----------
        n_layers : int value specifying the number of layers

        layer_sizes : integer array of size n_layers specifying the number of nodes in each layer

        activation : string specifying the activation function to be used
                     possible inputs: relu, sigmoid, linear, tanh

        learning_rate : float value specifying the learning rate to be used

        weight_init : string specifying the weight initialization function to be used
                      possible inputs: zero, random, normal

        batch_size : int value specifying the batch size to be used

        num_epochs : int value specifying the number of epochs to be used
        """
        self.n_layers = 0  #layers
        self.layer_sizes = []  #layer sizes
        self.activation = 'relu' #activation function
        self.learning_rate = 0.01 #learning rate
        self.weight_init = 'random' #weight assignment
        self.batch_size = 100 #batch size
        self.num_epochs = 10 #iterations
        self.acti_fns = ['relu', 'sigmoid', 'linear', 'tanh', 'softmax'] # activation functions
        self.weight_inits = ['zero', 'random', 'normal'] # weights assignment function
        self.W = [] # weights assignment
        self.B = [] # bias assignment
        self.losses_train= [] # array to store losses for train data
        self.accuracies_train = [] # array to store accuracies for train data
        self.losses_test= [] # array to store losses for test data
        self.accuracies_test = [] # array to store accuracies for test data
        self.fun = None # function to be applied in layers
        self.fun_grad = None #function gradient to be applied in fun_grad

        #checking for exception 
        if activation not in self.acti_fns:
            raise Exception('Incorrect Activation Function')

        if weight_init not in self.weight_inits:
            raise Exception('Incorrect Weight Initialization Function')
        self.n_layers = n_layers
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.weight_init = weight_init
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.activation = activation

        #checking if no.of layers = length of the array of layer sizes
        if (n_layers != len(layer_sizes)):
            raise Exception('Wrong parameters layers and their sizes')


    #RELU function for single variable in order to vectorize
    def RELU(self, a):
        return max(0.0, a)

    #RELU diff for single variable
    def RELU_diff(self, a):
        if a > 0:
            return 1.0
        else:
            return 0.0

    def relu(self, X):
        """
        Calculating the ReLU activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        # f = np.vectorize(self.RELU)
        # return f(X)
        return (X>0)*X

    def relu_grad(self, X):
        """
        Calculating the gradient of ReLU activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        # f = np.vectorize(self.RELU_diff)
        # return f(X)
        return (X>0)*1

    #sigmoid function for single variable
    def sig(self, a):
        return 1/(1+math.exp(-1*a))
    #sigmoid gradient
    def sig_diff(self, a):
        return self.sig(a)*(1-self.sig(a))

    def sigmoid(self, X):
        """
        Calculating the Sigmoid activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        # f = np.vectorize(self.sig)
        # return f(X)
        return 1.0 / (1+np.exp(-X))

    def sigmoid_grad(self, X):
        """
        Calculating the gradient of Sigmoid activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        # f = np.vectorize(self.sig_diff)
        # return f(X)

        sigma = self.sigmoid(X)
        return sigma*(1-sigma)

    def linear(self, X):
        """
        Calculating the Linear activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        X = np.clip(X,-1000,1000)
        return X

    def linear_grad(self, X):
        """
        Calculating the gradient of Linear activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return np.ones((X.shape[0],X.shape[1]))

    #tanh for a single variable
    def tan(self, a):
        return (math.exp(a)-math.exp(-1*a))/(math.exp(a)+math.exp(-1*a))
    #tanh gradient for single variable
    def tan_diff(self, a):
        return 1 - (self.tan(a))**2

    def tanh(self, X):
        """
        Calculating the Tanh activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return np.tanh(X)

    def tanh_grad(self, X):
        """
        Calculating the gradient of Tanh activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return 1- (self.tanh(X))**2

    def softmax(self, X):
        """
        Calculating the ReLU activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        if (self.activation == 'linear'):
        	X = np.clip(X,-1000,1000)
        exps = np.exp(X-X.max())
        return exps / np.sum(exps,axis=1)[:,None]

    def softmax_grad(self, X):
        """
        Calculating the gradient of Softmax activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return self.softmax(X)*(1-self.softmax(X))

    def zero_init(self, shape):
        """
        Calculating the initial weights after Zero Activation for a particular layer

        Parameters
        ----------
        shape : tuple specifying the shape of the layer for which weights have to be generated

        Returns
        -------
        weight : 1-dimensional numpy array which contains the initial weights for the requested layer
        """
        return np.zeroes(shape)

    def random_init(self, shape):
        """
        Calculating the initial weights after Random Activation for a particular layer

        Parameters
        ----------
        shape : tuple specifying the shape of the layer for which weights have to be generated

        Returns
        -------
        weight : 1-dimensional numpy array which contains the initial weights for the requested layer
        """
        return np.random.rand(shape[0],shape[1]) * 0.01

    def normal_init(self, shape):
        """
        Calculating the initial weights after Normal(0,1) Activation for a particular layer

        Parameters
        ----------
        shape : tuple specifying the shape of the layer for which weights have to be generated

        Returns
        -------
        weight : 1-dimensional numpy array which contains the initial weights for the requested layer
        """
        return np.random.normal(scale=1,size=(shape[0],shape[1])) * 0.01

    def fit(self, X, y,test_X =[],test_Y= []):
        """
        Fitting (training) the linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as training data.

        y : 1-dimensional numpy array of shape (n_samples,) which acts as training labels.

        Returns
        -------
        self : an instance of self
        """

        # fit function has to return an instance of itself or else it won't work with test.py
        # train_X,test_X,train_Y,test_Y = train_test_split(X, y, test_size=0.2, stratify=y)

        #checking what weights to assign based on weight init in fit function
        if self.weight_init == 'normal':
            for i in range(self.n_layers-1):
                self.W.append(self.normal_init((self.layer_sizes[i], self.layer_sizes[i+1])))
                self.B.append(self.normal_init((1,self.layer_sizes[i+1])))
        elif self.weight_init == 'random':
            for i in range(self.n_layers-1):
                self.W.append(self.random_init((self.layer_sizes[i], self.layer_sizes[i+1])))
                self.B.append(self.random_init((1,self.layer_sizes[i+1])))
        elif self.weight_init == 'zero':
            for i in range(self.n_layers-1):
                self.W.append(self.zero_init((self.layer_sizes[i], self.layer_sizes[i+1])))
                self.B.append(self.zero_init((1,self.layer_sizes[i+1])))

        #checking for whta actiavtion function to use based on actiavtion value
        if self.activation == 'relu':
            self.fun = self.relu
            self.fun_grad = self.relu_grad  
        if self.activation == 'sigmoid':
            self.fun = self.sigmoid
            self.fun_grad = self.sigmoid_grad
        if self.activation == 'tanh':
            self.fun = self.tanh
            self.fun_grad = self.tanh_grad
        if self.activation == 'linear':
            self.fun = self.linear
            self.fun_grad = self.linear_grad
        if self.activation == 'softmax':
            self.fun = self.softmax
            self.fun_grad = self.softmax_grad
        #fitting the mode;
        for i in range(self.num_epochs):
            print(i)
            # applyng BGD(batch gradienet descent)
            for j in range(max(len(train_X)//self.batch_size,1)):
                print(j)
                z, a = self.forward_propogation(train_X[j*self.batch_size:(j+1)*self.batch_size],self.fun) # forward propogation
                self.backward_propogation(z, a, train_X[j*self.batch_size:(j+1)*self.batch_size], train_Y[j*self.batch_size:(j+1)*self.batch_size],self.fun_grad) #backward propogation
            #calculating accuracya nd losses, appending them to array to store later for graphs
            self.accuracies_train.append(self.score(train_X, train_Y))
            self.losses_train.append(self.log_loss(train_X,train_Y))
            self.accuracies_test.append(self.score(test_X, test_Y))
            self.losses_test.append(self.log_loss(test_X,test_Y))

            print(self.accuracies_train)
            print(self.accuracies_test)
            print(self.losses_train)
            print(self.losses_test)
        return self

    def predict_proba(self, X):
        """
        Predicting probabilities using the trained linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

        Returns
        -------
        y : 2-dimensional numpy array of shape (n_samples, n_classes) which contains the 
            class wise prediction probabilities.
        """

        # return the numpy array y which contains the predicted values
        z,a = self.forward_propogation(X,self.fun)
        return a[-1]

    def predict(self, X):
        """
        Predicting values using the trained linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

        Returns
        -------
        y : 1-dimensional numpy array of shape (n_samples,) which contains the predicted values.
        """

        # return the numpy array y which contains the predicted values
        z, a = self.forward_propogation(X,self.fun)
        prediction = np.argmax(a[-1],axis=1)
        return prediction

    def log_loss(self,X,y):
        '''
    	calculating entropy loss based on formula yi.log(a)

    	Paramters
    	------------
    	Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.
		y : 1-dimensional numpy array of shape (n_samples,) which contains the predicted values.
        Returns
        -------
        float value loss
        '''
        loss = 0
        z, a = self.forward_propogation(X,self.fun)
        print(a[-1])
        loss = -1*np.mean(y*np.log(a[-1]+10**(-10)))
        return loss

    def log_loss_2(self,X,y):

        '''
    	calculating entropy loss based on formula yi.log(ai)+(1-yi)log(1-ai)

    	Paramters
    	------------
    	Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.
		y : 1-dimensional numpy array of shape (n_samples,) which contains the predicted values.
        Returns
        -------
        float value loss
        '''
        loss = 0
        z, a = self.forward_propogation(X,self.fun)
        print(a[-1])

        loss = -1*np.sum(np.multiply(np.log(a[-1]+10**(-10)),y) + np.multiply(np.log(1-a[-1]+10**(-10)),1-y),axis=1)
        return np.mean(loss)


    def score(self, X, y):
        """
        Predicting values using the trained linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

        y : 1-dimensional numpy array of shape (n_samples,) which acts as testing labels.

        Returns
        -------
        acc : float value specifying the accuracy of the model on the provided testing set
        """

        # return the numpy array y which contains the predicted values
        prediction = self.predict(X)
        # print(y)
        count = 0
        print(prediction,prediction.shape)
        for i in range(len(prediction)):
            if y[i][prediction[i]]==1:
                count += 1
        return count/len(prediction)*100

    def forward_propogation(self, train_X,func):
        '''
    	calculate forward propogation for data

    	Parameters
        ----------
        train_X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as training data.
        func: activation function

        Returns
        -------
        z : output before activation layer
        a: output after activation layer

        '''
        Z = []
        A = []
        #applying layers
        for i in range(0, self.n_layers-1):
            if i == 0:
                Z.append(np.dot(train_X,self.W[i])+self.B[i])
                A.append(func(Z[-1]))
            elif i != self.n_layers-2:
                Z.append(np.dot(A[i-1],self.W[i])+self.B[i])
                A.append(func(Z[-1]))
            else:
                Z.append(np.dot(A[i-1],self.W[i])+self.B[i])
                A.append(self.softmax(Z[-1]))
                # print(Z[-1])
                # print(A[-1])
                # print(np.sum(A[-1],axis=1))

        return Z, A

    def backward_propogation(self, Z, A, train_X, train_Y ,func_grad):
        '''
    	backward propgation and update of weights

		Parameters
        ----------
        train_X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as training data.
        train_Y: 2-dimensional numpy array of shape (n_samples, n_classes) which acts as training data.
		Z: output from forward propogation
		A: output from forward propgation
		func_grad: Gradient function
		
		returns
		----------
		null
		'''
		
        n = len(train_Y)
        Grad_W = []
        Grad_B = []
        for i in range(self.n_layers-2, -1, -1):
            if (i == self.n_layers-2):
                dZ2 = A[i]-train_Y
                Grad_W.insert(0, np.dot(A[i-1].T,dZ2)/n)
                Grad_B.insert(0, np.sum(dZ2, axis=0)/n)
            elif (i != 0):
                dZ = np.multiply(np.dot(dZ2,self.W[i+1].T), func_grad(Z[i]))
                Grad_W.insert(0, np.dot(A[i-1].T,dZ)/n)
                Grad_B.insert(0, np.sum(dZ, axis=0)/n)
                dZ2 = dZ
            else:
                dZ = np.multiply(np.dot(dZ2,self.W[i+1].T), func_grad(Z[i]))
                Grad_W.insert(0, np.dot(train_X.T,dZ)/n)
                Grad_B.insert(0, np.sum(dZ, axis=0)/n)
        for i in range(0, self.n_layers-1):
            self.W[i] = self.W[i]-self.learning_rate*Grad_W[i]
            self.B[i] = self.B[i]-self.learning_rate*Grad_B[i]
    def print_loss_and_acc(self):
        '''
    	function to print loss and accuracy
    	Parameters
    	---------
    	None

    	returns
    	---------
    	None
    	'''
        print("model = ",self.activation)
        print("train_loss = ",self.losses_train[-1])
        print("test_loss = ",self.losses_test[-1])
        print("train_accuracy = ",self.accuracies_train[-1])
        print("test_accuracy = ",self.accuracies_test[-1])
