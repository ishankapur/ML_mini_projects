import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class MyPreProcessor():
    """
    My steps for pre-processing for the three datasets.
    """

    def __init__(self):
        pass

    def pre_process(self, dataset):
        """
        Reading the file and preprocessing the input and output.
        Note that you will encode any string value and/or remove empty entries in this function only.
        Further any pre processing steps have to be performed in this function too. 

        Parameters
        ----------

        dataset : integer with acceptable values 0, 1, or 2
        0 -> Abalone Dataset
        1 -> VideoGame Dataset
        2 -> BankNote Authentication Dataset
        
        Returns
        -------
        X : 2-dimensional numpy array of shape (n_samples, n_features)
        y : 1-dimensional numpy array of shape (n_samples,)
        """
        np_array=[]

        # np.empty creates an empty array only. You have to replace this with your code.
        if dataset == 0:
            # Implement for the abalone dataset
            d = pd.read_csv("Dataset.data",sep=" ",header=None)
            filtered_data = d.dropna(axis='columns', how='all')
            np_array = filtered_data.to_numpy()

            np_array=np_array[1:,1:]
            np_array=np_array.astype(np.float)

            np.random.shuffle(np_array)

        elif dataset == 1:
            # Implement for the video game dataset
            d = pd.read_csv("VideoGameDataset - Video_Games_Sales_as_at_22_Dec_2016.csv",usecols=[9,10,12],sep=",",header=None)
            filtered_data = d.dropna()
            filtered_data=filtered_data[filtered_data[12]!='tbd']
            np_array = filtered_data.to_numpy()

            np_array=np_array[1:,:]

            np_array=np_array.astype(np.float)
            np_array[:,[0,2]]=np_array[:,[2,0]]
            print(np_array)

        elif dataset == 2:
            # Implement for the banknote authentication dataset
            d = pd.read_csv("data_banknote_authentication.txt",sep=",",header=None)
            filtered_data = d.dropna(axis='columns', how='all')
            np_array = filtered_data.to_numpy()
        np.random.shuffle(np_array)
        np_array=np.c_[np.ones(len(np_array)),np_array]
        
        X=np_array[:,0:len(np_array[0])-1]
        Y=np_array[:,len(np_array[0])-1:]

        return X, Y

class MyLinearRegression():
    """
    My implementation of Linear Regression.

    """
    alpha=0
    epocs=0
    b=[]
    def __init__(self,alpha=0.6,epocs=1500):
        self.alpha=alpha
        self.epocs=epocs
    def j_theta_linear_RMSE(self,x,y,b):
        return math.sqrt(self.j_theta_linear_MSE(x,y,b))
    def j_theta_linear_MAE(self,x,y,b):
        t = b.dot(x.T)
        t = np.subtract(t,y.T)
        t = np.absolute(t)
        return np.sum(t)/len(y)
    def j_theta_linear_MSE(self,x,y,b):
        t = b.dot(x.T)
        t = np.subtract(t,y.T)
        t =np.square(t)
        return np.sum(t)/len(y)
    def j_theta_linear_dif_MAE(self,x,y,b):
        t = b.dot(x.T)
        t = np.subtract(t,y.T)
        t = np.sign(t)
        xj = t.dot(x)
        return xj/len(y)
    def J_theta_linear_dif_MSE(self,x,y,b):
        t = b.dot(x.T)
        t = np.subtract(t,y.T)
        xj = t.dot(x)
        return xj/len(y)
    def j_theta_dif_RMSE(self,x,y,b):
        # print(x.shape,b.shape)
        t = b.dot(x.T)
        # print(t.shape)
        t = np.subtract(t,y.T)
        xj = t.dot(x)
        # print(xj.shape)
        return xj/len(y)
    def gradient_desc_linear_RMSE(self,x,y,b,alpha):
        return b - alpha*self.j_theta_dif_RMSE(x,y,b)/self.j_theta_linear_RMSE(x,y,b)
    def gradient_desc_linear_MSE(self,x,y,b,alpha):
        return b - alpha*self.J_theta_linear_dif_MSE(x,y,b)
    def gradient_desc_linear_MAE(self,x,y,b,alpha):
        return b - alpha*self.j_theta_linear_dif_MAE(x,y,b)
    def k_fold_division(x,y,k,epocs,alpha):
        print(np_array.shape)
        fold=np.array_split(x,k)
        fold_y=np.array_split(y,k)
        iterations=[]
        train_loss =[0]*epocs
        valid_loss=[0]*epocs
        for i in range(k):
            X=fold.copy()
            test_X=X[i][:]
            X=np.concatenate(X[:i]+X[i+1:],axis=0)
            train_X = X
            
            b=[0]*(len(X[0]))
            b= np.asarray(b)
            
            Y=fold_y.copy()
            test_Y=Y[i][:]
            Y=np.concatenate(Y[:i]+Y[i+1:],axis=0)
            train_Y = Y
            for i in range(epocs):
                b = gradient_desc_linear_MSE(train_X,train_Y,b,alpha)
                train_loss[i]=train_loss[i]+j_theta_linear_MSE(train_X,train_Y,b)
                valid_loss[i]=valid_loss[i]+j_theta_linear_MSE(test_X,test_Y,b)
        for i in range(epocs):
            train_loss[i]=train_loss[i]/k
            valid_loss[i]=valid_loss[i]/k
            iterations.append(i+1)
        print(train_loss[epocs-1],valid_loss[epocs-1])
        #return train_loss[epocs-1],valid_loss[epocs-1]
        plt.plot(iterations,valid_loss,label="validation loss",alpha=0.6,linewidth=5,color="cyan")
        plt.plot(iterations,train_loss,label="training loss",color="magenta")
        plt.xlabel("no. of iterations")
        plt.ylabel("loss value")
        plt.title("loss for MAE 10 fold")
        plt.legend()
        plt.show()

    def normal_closed_form(x,y):
  		xtx = np.dot(x.T,x)
  		xty = np.dot(x.T,y)
  		xtx_inv = np.linalg.inv(xtx)
  		det = np.linalg.det(xtx)
  		return np.dot(xtx_inv,xty).T
    def fit(self, X, Y,option="MSE"):
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

        b=[0]*(len(X[0]))
        b= np.asarray(b)
        epocs= self.epocs
        if option=="MSE":
            print("MSE")
            for i in range(epocs):
                b = self.gradient_desc_linear_MSE(X,Y,b,self.alpha)
            print(self.j_theta_linear_MSE(X,Y,b))
        elif option=="RMSE":
            print("RMSE")
            for i in range(epocs):
                b = self.gradient_desc_linear_RMSE(X,Y,b,self.alpha)
            print(self.j_theta_linear_RMSE(X,Y,b))
        elif option=="MAE":
            print("MAE")
            for i in range(epocs):
                b = self.gradient_desc_linear_MAE(X,Y,b,self.alpha)
            print(self.j_theta_linear_MAE(X,Y,b))
        else:
            print("option not found, performing MSE as evaluation matrix")
            for i in range(epocs):
                b = self.gradient_desc_linear_MSE(X,Y,b,self.alpha)
            print(self.j_theta_linear_MSE(X,Y,b))

        self.b=b
        return self

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
        return self.b.dot(X.T)
    


class MyLogisticRegression():
    """
    My implementation of Logistic Regression.
    """

    alpha=0
    epocs=0
    b=[]
    def __init__(self,alpha=0.1,epocs=3000):
        self.alpha=alpha
        self.epocs=epocs

    
    def gradient_desc(self,x,y,b,alpha):
        new_b=[]
        for i in range(len(b)):
            new_b.append(b[i]-alpha*self.J_theta_dif(x,y,b,i))
        return new_b
    def J_theta_dif(self,x,y,b,j):
        s=0
        global hp
        if j==0:
            hp=[]
            for i in range(len(y)):
                l=self.hypotheis_logistic(b,x[i])-y[i]
                s=s+l*x[i][j]
                hp.append(l)
        else:
             for i in range(len(y)):
                 s=s+hp[i]*x[i][j]
        return s/len(y)
    def hypotheis_logistic(self,b,x):
        s=0
        for i in range(len(x)):
            s=s+(b[i]*x[i])
        l=1/(1+math.exp(-1*s))
        return l
    def accuracy_logtsic(self,b,x,y):
        tr=0
        for i in range(len(x)):
            pr=self.hypotheis_logistic(b,x[i])
            if ( (pr>=0.5 and y[i]==1) or (pr<=0.5 and y[i]==0)):
                tr+=1
        return (tr/len(x))*100
    def J_theta_logistic(self,x,y,b):
        s=0
        for i in range(len(y)):
            s=s+y[i]*math.log(self.hypotheis_logistic(b,x[i]))+(1-y[i])*math.log((1-self.hypotheis_logistic(b,x[i])))
        return -1*s/len(y)
    def scholastic_gradient_descent(self,x,y,b,alpha):
        new_b=[]
        for k in range(len(x)):
            for i in range(len(b)):
                new_b.append(b[i]-alpha*self.J_theta_dif_2(x,y,b,k,i))
            b=new_b
            new_b=[]
        return b
    def J_theta_dif_2(self,x,y,b,i,j):
        return (self.hypotheis_logistic(b,x[i])-y[i])*x[i][j]
    def fit(self, X, Y,option="BGD"):
        """
        Fitting (training) the logistic model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as training data.

        y : 1-dimensional numpy array of shape (n_samples,) which acts as training labels.
        
        Returns
        -------
        self : an instance of self
        """

        # fit function has to return an instance of itself or else it won't work with test.py
        b=[0]*(len(X[0]))
        epocs=self.epocs
        if (option=="SGD"):
            for i in range(epocs):
                i_d = np.random.choice(len(X), size=1)[0]
                b = self.scholastic_gradient_descent(X[i_d:i_d+1,:],Y[i_d:i_d+1,:],b,self.alpha)
        else:
            for i in range(epocs):
            
                b = self.gradient_desc(X,Y,b,self.alpha)
        print(self.J_theta_logistic(X,Y,b))
        self.b=b
        return self

    def predict(self, X):
        """
        Predicting values using the trained logistic model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

        Returns
        -------
        y : 1-dimensional numpy array of shape (n_samples,) which contains the predicted values.
        """

        # return the numpy array y which contains the predicted values
        Y=[0]*len(X)
        for i in range(len(X)):
            pr=self.hypotheis_logistic(self.b,X[i])
            if pr>=0.5:
                Y[i]=1
        return np.asarray(Y)

