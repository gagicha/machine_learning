#data is downloaded locally 

from sklearn import model_selection,tree, datasets
import numpy as np
import pandas as pd

df= pd.read_csv('breast-cancer-wisconsin.txt')
df.drop(['id'],1,inplace=True)
df.replace('?',-99999,inplace=True)

x=np.array(df.drop(['class'],1))
y=np.array(df['class'])

x_train,x_test,y_train,y_test= model_selection.train_test_split(x,y)

clf=tree.DecisionTreeClassifier()
clf.fit(x_train,y_train)
clf.score(x_test,y_test)

#accuracy comes around 93%


