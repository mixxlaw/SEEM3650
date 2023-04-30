import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


#read data from excel file
df = pd.read_excel('seem3650data.xlsx')


#define x and y-axis
x = df[['Unemployment_rate', 'log(GDP)', 'Police_force_count']].values    #independent variables
y = df['Overall_crime_rate'].values                                       #dependent variables


# Standardize the independent variables
scaler = StandardScaler()
X_std = scaler.fit_transform(x)


#split data into training set and testing set
x_train, x_test, y_train, y_test = train_test_split(X_std, y, test_size=0.3, random_state=42)


#fit the model on the training data
model = SVR(kernel = 'rbf')
model.fit(x_train, y_train)


#use the trained model to predict the output for the testing set
y_pred = model.predict(x_test)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('Mean Squared Error on testing set:', mse)
print('R-squared Score on testing set:', r2)


X_grid = np.arange(min(x[:,0]), max(x[:,0]), 0.01).reshape(-1,1)
plt.scatter(x_test[:,0], y_test, color = 'red', label='Testing Data')
plt.plot(X_grid, model.predict(scaler.transform(np.concatenate((X_grid, np.zeros((len(X_grid),2))), axis=1))), color = 'blue', label='SVR Model')
plt.title(f'Truth or Bluff (SVR)\nMean Squared Error: {mse:.2f}')
plt.xlabel('Unemployment Rate, Log(GDP), Police Force Count')
plt.ylabel('Overall Crime Rate')
plt.legend()
plt.show()