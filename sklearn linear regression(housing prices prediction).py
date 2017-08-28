
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

boston=load_boston()
boston
#creating dataframe
df_x = pd.DataFrame(boston.data,columns=boston.feature_names)
df_y = pd.DataFrame(boston.target)

df_x.describe()
#modelling
reg=linear_model.LinearRegression()
#train test split dat 
x_train,x_test,y_train,y_test= train_test_split(df_x,df_y,test_size=0.2,random_state=4)

#fitting the data into the model 
reg.fit(x_train,y_train)

reg.coef_
#predicting using test data
a=reg.predict(x_test)

#mean square error
np.mean((a-y_test)**2)

