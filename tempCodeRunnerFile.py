from sklearn import datasets
data = datasets.load_boston()

# split train dataset and test dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data.data,data.target, test_size=0.15)
print(x_train.shape)

from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.test_functions import Ishigami
import numpy as np


# random forest
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(max_depth=10, n_estimators=150)
rf_model.fit(x_train, y_train)

y_pred = rf_model.predict(x_test)
for t, p in zip(y_test[:10], y_pred[:10]):
    print("correct value:", t, ">>>prediction:", p, "difference:", t-p)

import matplotlib.pyplot as plt
plt.plot(x_test, y_test, 'b.')
plt.plot(x_test, y_pred, 'r.')
plt.show()
# model performance assessment
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
print("MSE:", mean_squared_error(y_pred, y_test))
print("MAE:", mean_absolute_error(y_pred, y_test))
print("R2:", r2_score(y_pred, y_test))

# MSE: 12.714120815597113
# MAE: 2.390203235806607
# R2: 0.8204623192165318