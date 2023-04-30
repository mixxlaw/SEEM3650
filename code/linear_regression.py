import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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


#create a linear regression model and fit it to the training data
model = LinearRegression()
model.fit(x_train, y_train)


#predict the target variable for the testing set
y_pred = model.predict(x_test)




#evaluate the performance of the model
print(f"Coefficients: {model.coef_}")
mse = mean_squared_error(y_test, y_pred)
print("Mean squared error: %.2f" % mse)
print('R-squared score: %.2f' % r2_score(y_test, y_pred))



#plot the predicted values against the actual values
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual crime rate')
plt.ylabel('Predicted crime rate')
plt.title('Actual crime rate vs. Predicted Crime Rate')
plt.show()


