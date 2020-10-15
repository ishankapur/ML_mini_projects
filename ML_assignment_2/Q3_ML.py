
#importing libraries
from copy import deepcopy
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
import pickle

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

def k_fold(k,model,train,ans,ans_train,dp=2):

  '''
  k fold- runs a k fold for a specific model as given in the params

  input

  k - represnt k for k fold ( integer)
  model - any model from sklearn which has capabilities of set param, .fit and .predict
  train - train data ( X an Y both added to maintain order for shuffling)
  ans - validation error for each fold as 2 d array
  ans_train - training error for each fold

  optional:
  dp=2 - depth if its descision tree ( default set 2 for every other model where depth deosnt make sense)
  
  output - none ( ans,ans_train are modified in place)
  '''
  fold=np.array_split(train,k)
  for i in range(k):
    tr_X=fold.copy()
    val=tr_X[i][:]
    tr_X=np.concatenate(tr_X[:i]+tr_X[i+1:],axis=0)
    train_X = tr_X[:,0:len(tr_X[0])-1]
    train_Y = tr_X[:,len(tr_X[0])-1:]
    val_X=val[:,0:len(tr_X[0])-1]
    val_Y =val[:,len(tr_X[0])-1:]
    clf = deepcopy(model)
    an= accuracy_3rd(clf,train_X,val_X,train_Y,val_Y)
    ans[dp-2][i] = an[1]
    ans_train[dp-2][i] = an[0]

def t_t_split(X,Y_l):
  '''
  train test split - splits data into 80 20 rato

  input

  X- input features ( nd array)

  Y_l - output features ( 1 darray ( NX1))

  output-

  train - input+output features ( 80%)
  test_X - input features for testing (20%)
  test_Y - output features for the same

  '''

  np_arr = np.insert(X,len(X[0]),Y_l,1)
  np.random.shuffle(np_arr)
  train = np_arr[:8*len(np_arr)//10,:]
  test = np_arr[8*len(np_arr)//10:,:]
  test_X = test[:,:test.shape[1]-1]
  test_Y = test[:,test.shape[1]-1:]

  return train,test_X,test_Y

def GNB_part(train):
  '''
  calls GNB for k = 4 folds and report validation accuracies

  input

  train - X&Y combined

  output- none 
  prints the validation accuracies
  '''
  k=4
  ans = np.empty((max_depth-2,k))
  ans_train = np.empty((max_depth-2,k))

  model = GaussianNB()
  k_fold(k,model,train,ans,ans_train)

  print("validation accuracies" ,**ans)

def accuracy_3rd(model,X_train,X_val,Y_train,Y_val):
  '''
  fits and predict any model and report training and testing accuracy
  '''
  model.fit(X_train,Y_train.T[0])
  ypred =model.predict(X_train)
  from sklearn.metrics import accuracy_score
  
  a  =accuracy_score(Y_train,ypred)
  ypred =model.predict(X_val)
  
  return a*100,accuracy_score(Y_val,ypred)*100

def Gridsearch_with_sklearn(train,max_depth=20):

  '''
  calls sklearn descision tree classifier for k = 4  from 2 to maxdepth folds and report validation accuracies

  input

  train - X&Y combined

  optional - max_depth ( the max limit upto which iteration has to be done )default set to 20

  output- 
  optimal depth- which gave the max avg validaton error
  ans - 2 d array of size( max_depth-2 X k for k fold) reports validation accuracy for each max_depth from 2 to max_depth
  ans-train- report training accuracy
  depth- array which return depth use ( for plotting the graph {optional}) 
  prints the validation accuracies
  '''
  k=4
  depths=[]
  ans = np.empty((max_depth-2,k))
  ans_train = np.empty((max_depth-2,k))

  for dp in range(2,max_depth):
    depths.append(dp)
    model = tree.DecisionTreeClassifier(max_depth=dp)
    k_fold(k,model,train,ans,ans_train,dp)

  print("validation accuracies" ,*np.mean(ans,axis=1))
  max_v = np.argmax(np.mean(ans,axis=1))
  print("optimal depth = ", max_v+2)
  return max_v+2,ans,ans_train,depths

def plot_depthVsVal(ans,ans_train,depths,dset="A"):
  '''
  plot the graph of training and validation vs depth
  '''
  plt.plot(depths,np.mean(ans,axis=1),label="validation loss",linewidth=3,color="cyan")
  plt.plot(depths,np.mean(ans_train,axis=1),label="training loss",linewidth=3,color="red")
  plt.xlabel("max depth")
  plt.ylabel("loss value")
  plt.title("depths vs valid loss dataset "+dset)
  plt.legend()
  plt.show()

def save_model(optimal_depth,dset="A"):
  '''
  saves the model using pickle
  '''
  clf = tree.DecisionTreeClassifier(max_depth=optimal_depth)
  clf.fit(train[:,0:len(train[0])-1],train[:,len(train[0])-1:],)
  f = "weights/best_model "+str(dset)+".sav"
  pickle.dump(clf,open(f,'wb'))
  print("model saved successfully")

def load_model_and_predict(test_X,test_Y,dset="A"):
  '''
  loads the model using pickle
  '''
  f = "weights/best_model "+str(dset)+".sav"
  c = pickle.load(open(f,'rb'))
  r= c.score(test_X,test_Y)
  ypred = c.predict(test_X)
  print(r)
  return c, ypred
def confusion_matrix(y_pred,y_actual,no_of_classes):
  ''' returns confusion matrix for given no. of classes'''
  c_m = np.zeros((no_of_classes,no_of_classes),dtype=int)
  for i in range(len(y_pred)):
    c_m[y_pred[i]][y_actual[i]]+=1
  return c_m

def evaluation_report(y_pred,y_actual,no_of_classes):
  ''' same fucntion as classifictaion report of sklearn, report precision recall accuracy f score and macro avg'''
  c_m =confusion_matrix(y_pred,y_actual,no_of_classes)
  print("CONFUSION MATRIX")
  print(c_m)
  precision = []
  recall = []
  F1_score =[]
  acc = 0
  weighted_avg = 0
  for i in range(no_of_classes):
    precision.append(c_m[i][i]/np.sum(c_m,axis=0)[i])
    recall.append(c_m[i][i]/np.sum(c_m,axis=1)[i])
    F1_score.append(2*precision[-1]*recall[-1]/(precision[-1]+recall[-1]))
    acc+=c_m[i][i]
  acc=acc/np.sum(np.sum(c_m))
  print("\t SKLEARN EVALUATION METRIC")
  print("precision\t recall \t F1_score")
  for i in range(no_of_classes):
    print(round(precision[i],4),round(recall[i],4),round(F1_score[i],4),sep="\t\t")
  

  print(round(np.asarray(precision).mean(),4),round(np.asarray(recall).mean(),4),round(np.asarray(F1_score).mean(),4),sep="\t\t")
  print("accuracy = ",acc*100)

def generate_confusion_matrix_multi(probab,y_actual,no,threshold):
  '''
  generate confusion matrix with the help of a threshld given in the params
  '''
  conf = np.zeros((2,2))
  for i in range(len(probab)):
    if (probab[i][no]>=threshold and y_actual[i]==no):
      conf[0][0]+=1
    elif (probab[i][no]>=threshold and y_actual[i]!=no):
      conf[1][0]+=1
    elif (probab[i][no]<threshold and y_actual[i]==no):
      conf[0][1]+=1
    else:
      conf[1][1]+=1
  return conf

def PLOT_ROC(model,X_test,y_actual,no_of_classes,iter):
  '''plot ROC curve for a model,

  plz use no_of classes =1 for binary data and for more than 2 classes no_of classes = how many you want to print'''
  probab = model.predict_proba(X_test)
  print(probab)
  a=[]
  if (no_of_classes==1):
    a = (probab[:,0:1].T).tolist()[0]
    a.sort()
  print(a)
  threshold=0
  np_array_TPR = np.empty((no_of_classes,iter))
  np_array_FPR = np.empty((no_of_classes,iter))
  micro = np.empty((2,iter))
  macro = np.empty((2,iter))
  
  for j in range(iter):
    final_matrix = np.zeros((2,2))
    for i in range(no_of_classes):
      c_m = generate_confusion_matrix_multi(probab,y_actual,i,threshold)
      np_array_TPR[i][j] = c_m[0][0]/(c_m[0][0]+c_m[0][1])
      np_array_FPR[i][j] = c_m[1][0]/(c_m[1][0]+c_m[1][1])
      final_matrix = final_matrix+c_m
    if (no_of_classes==1):
      threshold = a[j*len(a)//iter]
    else:
      threshold+=0.05
      micro[1][j] = final_matrix[0][0]/(final_matrix[0][0]+final_matrix[0][1])
      micro[0][j] = final_matrix[1][0]/(final_matrix[1][0]+final_matrix[1][1])
      macro[1][j] = np.mean(np_array_TPR[:,j:j+1],axis=0)
      macro[0][j] = np.mean(np_array_FPR[:,j:j+1],axis=0)
  for i in range(no_of_classes):
    x = np_array_FPR[i].tolist()
    x= [0]+x+[1]
    y = np_array_TPR[i].tolist()
    y= [0]+y+[1]
    x.sort()
    y.sort()
    plt.plot(x,y,label="class"+str(i),linewidth=3)
  
  if (no_of_classes!=1):
    x_micro = micro[0].tolist()+[0]+[1]
    x_micro.sort()
    x_macro = macro[0].tolist()+[0]+[1]
    x_macro.sort()
    y_micro = micro[1].tolist()+[0]+[1]
    y_micro.sort()
    y_macro = macro[1].tolist()+[0]+[1]
    y_macro.sort()
    plt.plot(x_micro,y_micro,':',label="micro",linewidth=3)
    plt.plot(x_macro,y_macro,':',label="macro",linewidth=3)

  plt.plot([0,1],[0,1],'--',label="y=x",linewidth=1,color="black")
  
  plt.xlabel("FPR")
  plt.ylabel("TPR")
  plt.title("ROC B")
  plt.legend()
  plt.show()


dz= h5py.File('part_B_train.h5','r')
X,Y = preprocess(dz)
train,test_X,test_Y = t_t_split(X,Y)
opt_depth,ans,ans_train,depths=Gridsearch_with_sklearn(train,50)
plot_depthVsVal(ans,ans_train,depths,"B")
save_model(opt_depth,"B")
model,pred=load_model_and_predict(test_X,test_Y,"B")

evaluation_report(pred,test_Y.T[0],2)
PLOT_ROC(model,test_X,test_Y.T[0],1,40)





