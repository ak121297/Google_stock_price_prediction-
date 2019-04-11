#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 18:11:10 2019

@author: akashg
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
trainingset=pd.read_csv('Google_Stock_Price_Train.csv')

trainingset=trainingset.iloc[:,1:2].values
#we want two dimensional nummpy array thats why here we r using[...1:2]instead of simply [1]e

from sklearn.preprocessing import MinMaxScaler
ad=MinMaxScaler()
trainingset=ad.fit_transform(trainingset)
xtrain=trainingset[0:1257]
ytrain=trainingset[1:1258]

#splitting
#from sklearn.model_selection import train_test_split
#xtrain,xtest,ytrain,ytest = train_test_split(x,y, test_size= )

#2d-->3d array
xtrain=np.reshape(xtrain,(1257,1,1))

#building rnn
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

regressor=Sequential()
regressor.add(LSTM(units=4,activation='sigmoid',input_shape=(None,1)))
regressor.add(Dense(units=1))

regressor.compile(optimizer='adam',loss='mean_squared_error')

regressor.fit(xtrain,ytrain,batch_size=32,epochs=300)

testset=pd.read_csv('Google_Stock_Price_Test.csv')
google_pred=testset.iloc[:,1:2].values 
google_pred2=google_pred
google_pred2=ad.transform(google_pred2)
google_pred2=np.reshape(google_pred2,(20,1,1))

final_pred=regressor.predict(google_pred2)
final_pred=ad.inverse_transform(final_pred)

plt.plot(google_pred,color='red',label='real price of stock')

plt.plot(final_pred,color='red',label='predicted price of stock')
plt.title('Google stock price prediction')
plt.xlabel('time')
plt.ylabel('google stock price')
plt.legend()
plt.show()
