import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score


#read data from excel file
df = pd.read_excel('seem3650data.xlsx')


#define x and y-axis
x = df[['Unemployment_rate', 'log(GDP)', 'Police_force_count']]    #independent variables
y = df['Overall_crime_rate']                                       #dependent variables

#standardize the independent variables
scaler = StandardScaler()
X_std = scaler.fit_transform(x)


#define the range of lambda values to test
lambdas = np.logspace(-3, 3, 100)


#compute the mean squared error for each value of lambda using cross-validation
mse = []
for lambda_val in lambdas:
    model = LassoCV(alphas=[lambda_val], cv=5)
    scores = cross_val_score(model, X_std, y, scoring='neg_mean_squared_error', cv=5)
    mse.append(-np.mean(scores))



#plot the mean squared error as a function of lambda
plt.plot(lambdas, mse)
plt.xscale('log')
plt.xlabel('log(Lambda)')
plt.ylabel('Mean Squared Error')
plt.title('Mean Squared Error as a Function of Lambda')
plt.show()


#fit the LASSO regression model with the optimal lambda value
optimal_lambda = lambdas[np.argmin(mse)]
model = LassoCV(alphas=[optimal_lambda], cv=5).fit(X_std, y)

#print the coefficients and the selected features
print(f"Coefficients: {model.coef_}")
print(f"Selected features: {x.columns[model.coef_ != 0].tolist()}")