
import matplotlib.pyplot as plt
from sklearn import datasets
#support vector machines
from sklearn import svm

digits=datasets.load_digits()
print(digits)
#svm classifier for UCI handwriteen digit classification
#when gamma is low accuracy is good, when its large accuracy not good
clf= svm.SVC(gamma=0.001,C=100)
#leaving out last 10 columns 
x,y= digits.data[:-10],digits.target[:-10]
clf.fit(x,y)
#testing the data on the last 10 digits which was not used for training
print(clf.predict(digits.data[-6]))

plt.imshow(digits.images[-6])
plt.show()

