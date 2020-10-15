from scratch import MyLinearRegression, MyLogisticRegression, MyPreProcessor
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

preprocessor = MyPreProcessor()

print('Linear Regression')

X, y = preprocessor.pre_process(1)

# Create your k-fold splits or train-val-test splits as required



linear = MyLinearRegression(alpha=0.0001)
linear.fit(X, y,option="RMSE")
ypred = linear.predict(X)

print('Predicted Values:', ypred)
# print('True Values:', ytest)

print('Logistic Regression')

X, y = preprocessor.pre_process(2)

# Create your k-fold splits or train-val-test splits as required

logistic = MyLogisticRegression(epocs=50000,alpha=0.01)
logistic.fit(X, y,option="SGD")

ypred = logistic.predict(X)

print('Predicted Values:', ypred)
# print('True Values:', ytest)