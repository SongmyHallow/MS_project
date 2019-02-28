import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
 
# base test dataset 
n_dots = 400
X = 5 * np.random.rand(n_dots, 1)
y = np.exp(X).ravel()
 
#add noise
y += 0.1 * np.random.rand(n_dots) - 0.1
# plt.plot(X,y, 'r.')
# plt.show()

#KNN Regression
k = 2
knn = KNeighborsRegressor(k)
knn.fit(X,y)
prec = knn.score(X, y)  #计算拟合曲线针对训练样本的拟合准确性
print(prec)
 
#generate enough predict data
T = np.linspace(0, 5, 500)[:, np.newaxis]
y_pred = knn.predict(T)
 
#draw regress curve
plt.figure(figsize=(16, 10), dpi = 144)
plt.scatter(X, y, c='g', label='data', s=100) #训练样本
plt.scatter(T, y_pred, c='k', label='prediction', lw=2) #拟合曲线
plt.axis('tight')
plt.title('KNN regression (k =%i)'%k)
# plt.title()
# plt.show()
# #print(X)
# #print(y)
 
plt.show()
