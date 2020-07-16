import pandas as pd
import numpy as np
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import pickle 

iqdata = pd.read_csv('iq_data.csv')  
X = iqdata[['PhyAge','MentalAge']]
y = iqdata['IQ']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
lm = LinearRegression()
lm.fit(X_train,y_train)
Pkl_Filename = "linear_model.pkl"  
with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(lm, file)