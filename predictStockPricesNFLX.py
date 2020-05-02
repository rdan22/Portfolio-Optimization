#Install the dependencies
import numpy as np
import pandas as pd 
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
plt.style.use('bmh')

#Store the data into a data frame
df = pd.read_csv('NFLX.csv') #April 25th, 2019 - 2020
print(df.head(6))

#Get the number of trading days
print(df.shape)

#visualize the close price data
plt.figure(figsize=(16,8))
plt.title('Netflix')
plt.xlabel('Days')
plt.ylabel('Close Price USD ($)')
plt.plot(df['Close'])
plt.show()

#Get the close price
df = df[['Close']]
print(df.head(4))

#Create a variable to predict the 'x' days out into the future
future_days = 25

#Create a new column (target) shifted 'x' units/days up
df['Prediction'] = df[['Close']].shift(-future_days)
print(df.tail(4))

#Create the feature data set (X) and convert to np array and remove the
#last 'x' rows/days
X = np.array(df.drop(['Prediction'], 1))[:-future_days]
print(X)

#Create the target data set (y) and convert it to np array and get all
#of the target values except the last 'x' rows/days 
y = np.array(df['Prediction'])[:-future_days]

#Split the data into 75% training and 25% testing
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

#Create the models
#Create the decision tree regressor model
tree = DecisionTreeRegressor().fit(x_train, y_train)
#Create the linear regression model
lr = LinearRegression().fit(x_train, y_train)

#Get the last 'x' rows of the feature data set
x_future = df.drop(['Prediction'], 1)[:-future_days]
x_future = x_future.tail(future_days)

#Convert to numpy array
x_future = np.array(x_future)
print(x_future)

#Show the model tree prediction 
tree_prediction = tree.predict(x_future)
print(tree_prediction)
print()
#Show the model linear regression prediction
lr_prediction = lr.predict(x_future)
print(lr_prediction)

#Visualize the data (Tree)
predictions = tree_prediction
valid = df[X.shape[0]:]
valid['Predictions'] = predictions
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Days')
plt.ylabel('Close Price USD ($)')
plt.plot(df['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Original Data', 'Validation Data', 'Prediction Data'])
plt.show()

#Visualize the data (Linear)
predictions = lr_prediction
valid = df[X.shape[0]:]
valid['Predictions'] = predictions
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Days')
plt.ylabel('Close Price USD ($)')
plt.plot(df['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Original Data', 'Validation Data', 'Prediction Data'])
plt.show()
