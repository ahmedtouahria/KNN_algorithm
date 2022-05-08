'''
K nearest neighbour (KNN) Algorithm classification
'''

'''  import necessery library  '''
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# load iris data from datasets
iris=datasets.load_iris()

# splite iris.data,iris.target to x and y
x , y =iris.data,iris.target

x_train ,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
print(x_train.shape)
print(x_train[0])

from matplotlib.colors import ListedColormap
colormap=ListedColormap(['r','g','b'])
plt.figure()
plt.scatter(x[:,0],x[:,1],c=y,cmap=colormap)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(x,y)
pred = knn.predict(x_test)
print(pred)

def accu(y_test,pred):
    acc=np.sum(y_test==pred)/len(y_test)
    return acc
print("acc = ",accu(y_test,pred)*100,"%")
 
