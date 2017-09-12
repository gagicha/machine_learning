import pandas as pd
import numpy as np
from sklearn import model_selection, linear_model
#to the uci dataset we add the attributes row and then use that in this example.
df= pd.read_csv('dataset/prima-indians-diabetes.txt')

x= np.array(df.drop(['Class variable'],1))
y= np.array(df['Class variable'])

x_train, x_test, y_train, y_test= model_selection.train_test_split(x,y,test_size=0.2)

clf=linear_model.LogisticRegression(C=1e5)

clf.fit(x_train, y_train)

clf.score(x_test,y_test)
#print(accuracy)
#.818
