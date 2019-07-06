import alamopy
import math
import numpy as np
import matplotlib.pyplot as plt
import examples
from sklearn.model_selection import train_test_split

# X_train = [[-5], [-1], [-3], [3], [2], [1], [-4], [5]]

X_train = [-5,-1,-3,3,2,1,-4,5]
y_train = [25, 1, 9, 9, 4, 1, 16, 25]
X_test = [[-2], [0], [4]]
y_test = [4, 0, 16]
# X_train,X_test,y_train,y_test=train_test_split(xdata,ydata,test_size=0.25)
print(X_train,X_test,y_train,y_test)
res = alamopy.alamo(xdata=X_train,zdata=y_train,xval=X_test,zval=y_test,xmax=5,xmin=-5,monomialpower=(1,2),showalm=True)
print(res['model'])
