
# coding: utf-8

# In[6]:

#breast cancer classification(malignant/benign) using UCI breast cnacer wisconsin text data 
# https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/ 
import numpy as np
from sklearn import preprocessing, model_selection, neighbors
import pandas as pd
df = pd.read_csv('breast-cancer-wisconsin.txt')


# In[7]:

#remove the missing values represented by ?
df.replace('?',-99999,inplace=True)
#giving 99999 value tells the algo that this is an outlier
# or df.dropna(inplace=True)


# In[10]:

#remove useless data, in this case its id
df.drop(['id'],1,inplace=True)


# In[35]:

x = np.array(df.drop(['class'],1))
y = np.array(df['class'])


# In[36]:

x_train,x_test,y_train,y_test= model_selection.train_test_split(x,y,test_size=0.2)


# In[37]:

clf=neighbors.KNeighborsClassifier()
clf.fit(x_train, y_train)
#a=clf.predict(x_test)
accuracy=clf.score(x_test,y_test)
print(accuracy)
#accuracy comes around 96%
#if we dont drop the id column the accuracy comes around 56%


# In[ ]:




# In[ ]:



