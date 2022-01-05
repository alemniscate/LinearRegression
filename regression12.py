import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

class CustomLinearRegression:

    def __init__(self, *, fit_intercept=True):

        self.fit_intercept = fit_intercept
#        self.coefficient = ...
#        self.intercept = ...

    def fit(self, X, y):
        a = np.array(X)
        if self.fit_intercept:
            a = np.insert(a, 0, 1, axis=1)
        b = np.array(y)
        w = np.linalg.inv(a.T@a)@a.T@b
        if self.fit_intercept:
            self.coefficient = w[1:]
            self.intercept = w[0]
        else:
            self.coefficient = w
            self.intercept = np.zeros(y.size, 1)
        pass

    def predict(self, X):
        a = np.array(X)
        if self.fit_intercept:
            a = np.insert(a, 0, 1, axis=1)
        w = np.insert(self.coefficient, 0, self.intercept)
        return a@w
        pass

    def r2_score(self, y, yhat):
        return 1 - np.sum((y - yhat) ** 2) / np.sum((y - np.average(yhat)) ** 2)
        pass

    def rmse(self, y, yhat):
        return np.sqrt(self.mse(y, yhat))  
        pass

    def mse(self, y, yhat):
        return np.sum((y - yhat) ** 2) / y.size


data = {
'f1':[
2.31,
7.07,
7.07,
2.18,
2.18,
2.18,
7.87,
7.87,
7.87,
7.87],

'f2':[
65.2,
78.9,
61.1,
45.8,
54.2,
58.7,
96.1,
100,
85.9,
94.3],

'f3':[
15.3,
17.8,
17.8,
18.7,
18.7,
18.7,
15.2,
15.2,
15.2,
15.2],

'y':[
24,
21.6,
34.7,
33.4,
36.2,
28.7,
27.1,
16.5,
18.9,
15]

}

df = pd.DataFrame(data)
regCustom = CustomLinearRegression(fit_intercept=True)
regCustom.fit(df[['f1', 'f2', 'f3']], df['y'])
c_yhat = regCustom.predict(df[['f1', 'f2', 'f3']])
c_intercept = regCustom.intercept
c_coefficient = regCustom.coefficient
c_R2 = regCustom.r2_score(df['y'], c_yhat)
c_rmse = regCustom.rmse(df['y'], c_yhat)

regSci = LinearRegression(fit_intercept=True)
regSci.fit(df[['f1', 'f2', 'f3']], df['y'])
s_yhat = regSci.predict(df[['f1', 'f2', 'f3']])
s_intercept = regSci.intercept_
s_coefficient = regSci.coef_
s_R2 = r2_score(df['y'], s_yhat)
s_rmse = np.sqrt(mean_squared_error(df['y'], s_yhat))

result = {}
result['Intercept'] = s_intercept - c_intercept
result['Coefficient'] = s_coefficient - c_coefficient
result['R2'] = s_R2 - c_R2
result['RMSE'] = s_rmse - c_rmse
print(result)