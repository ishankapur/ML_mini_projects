import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def normal_closed_form(x,y):
  xtx = np.dot(x.T,x)
  xty = np.dot(x.T,y)
  xtx_inv = np.linalg.inv(xtx)
  det = np.linalg.det(xtx)
  return np.dot(xtx_inv,xty).T

def j_theta_linear_MSE(x,y,b):
  t = b.dot(x.T)
  t = np.subtract(t,y.T)
  t =np.square(t)
  return np.sum(t)/len(y)

def bias(y,y_test):
  y_bar = np.mean(y)
  yt = np.absolute(y_test-y_bar)
  return np.mean(yt)

def variance(y):
  return np.var(y,ddof=1)

def ypred_return(x,b):
  return b.dot(x.T).T

def net_bias_variance(y,y_test):
  print(y.shape)
  baises = np.empty(len(y))
  variances = np.empty(len(y))
  for i in range(len(y)):
    baises[i] = bias(y[i],y_test[i])
    variances[i] = variance(y[i])
  baises = np.absolute(baises)
  net_bias = np.mean(baises)
  net_variance = np.mean(variances)
  print("bias",net_bias,"variance",net_variance)
  return net_bias,net_variance

def bootstrap(x,iterations,size):
  
  mses = np.array([])
  np.random.shuffle(x)
  train = x[:8*len(x)//10,:]
  test = x[8*len(x)//10:,:]
  test_X = test[:,:test.shape[1]-1]
  test_Y = test[:,test.shape[1]-1:]
  test_X = np.c_[np.ones(len(test_X)),test_X]
  y_pred=np.empty((test_X.shape[0],0))
  print(test_X.shape)
  for i in range(iterations):
    id = np.random.choice(len(train),size,replace=True)
    np_arr = []
    for j in id:
      np_arr.append(train[j])
    np_arr=np.asarray(np_arr)
    X = np_arr[:,:np_arr.shape[1]-1]
    Y = np_arr[:,np_arr.shape[1]-1:]
    X = np.c_[np.ones(len(X)),X]
    b = normal_closed_form(X,Y)
    ms =j_theta_linear_MSE(test_X,test_Y,b)
    mses =np.append(mses,ms)
    zd = ypred_return(test_X,b)
    y_pred = np.insert(y_pred,len(y_pred[1]),zd.T[0],axis=1)
    
  bias,variance = net_bias_variance(y_pred,test_Y)
  print("AVG MSE",np.mean(mses))
  print("irreducible uncertainity",np.mean(mses)- bias**2 - variance)


d = pd.read_csv("weight-height.csv",sep=",",header=None)
d = d.dropna()
np_array = d.to_numpy()
np_array=np_array[1:,1:]
np_array=np_array.astype(np.float)

bootstrap(np_array,20,100)