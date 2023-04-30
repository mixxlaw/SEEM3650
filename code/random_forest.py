import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


#read data from excel file
df = pd.read_excel('seem3650data.xlsx')


#define x and y-axis
x = df[['Unemployment_rate', 'log(GDP)', 'Police_force_count']]    #independent variables
y = df['Overall_crime_rate']                                       #dependent variables


# Standardize the independent variables
scaler = StandardScaler()
X_std = scaler.fit_transform(x)


#split data into training set and testing set
x_train, x_test, y_train, y_test = train_test_split(X_std, y, test_size=0.3, random_state=42)


#define the Random Forest Regression model and set the hyperparameters
rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)


#fit the model on the training data
rf.fit(x_train, y_train)


#use the trained model to predict the output for the testing set
y_pred = rf.predict(x_test)


#evaluate the model performance on the testing set
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('Mean Squared Error on testing set:', mse)
print('R-squared Score on testing set:', r2)



#plot the MSE with respect to actual values and predicted valeus
plt.plot([0, max(y_test)], [0, max(y_test)], 'k--', lw=2)
plt.scatter(y_test, y_pred)
plt.xlabel('Actual crime rate')
plt.ylabel('Predicted crime rate')
plt.title('Mean Squared Error: {:.2f}'.format(mse))
plt.show()