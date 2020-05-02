#Description: Creating an artificial neural network (LSTM) to predict
#			  the closing stock price of a corporation (Apple) using the past 60 day stock price. 
#Use: Python 3

import math 
import pandas_datareader as web
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler 
from keras.models import Sequential
from keras.layers import Dense, LSTM 
import matplotlib.pyplot as plt 
plt.style.use('fivethirtyeight')

#Get the stock quote 
df = web.DataReader('AAPL', data_source = 'yahoo', start = '2012-01-01', end='2020-04-01')
#Show the data 
print(df)

#Get the number of rows and columns in the data set
print(df.shape)	#(2075, 6)

#Visualize the closing price history
plt.figure(figsize = (16, 8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize = 18)
plt.ylabel('Close Price USD ($)', fontsize = 18)
plt.show()

#Create a new dataframe with only the Close column
data = df.filter(['Close'])
#Convert the dataframe to a numpy array
dataset = data.values 
#Get the number of rows to train the model on 
training_data_len = math.ceil(len(dataset) * 0.8)
print(training_data_len)

#Scale the data 
#Nearly always advantageous to apply preprocessing transformations (scaling, normalization, etc.) to the
#input data before it is presented to an ANN
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)
print(scaled_data)

#Create the training data set
#Create the scaled training data set 
train_data = scaled_data[0:training_data_len, :]
#Split the data into x_train and y_train data sets 
x_train = []
y_train = []

for i in range(60, len(train_data)):
	#append to x_train data set
	x_train.append(train_data[i-60:i, 0])
	y_train.append(train_data[i, 0])
	if i <= 60:
		print(x_train)
		print(y_train)
		print()

#Convert the x_train and y_train to numpy arrays 
x_train, y_train = np.array(x_train), np.array(y_train)

#Reshape the data
#Only one feature = closing price 
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
print(x_train.shape)

#build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
#Second layer
model.add(LSTM(50, return_sequences = False))
#regular densely connected neural layer with 25 neurons
model.add(Dense(25))
model.add(Dense(1))

#Compile the model
#Loss function determines how well the model did in training
model.compile(optimizer = 'adam', loss= 'mean_squared_error')

#Train the model
model.fit(x_train, y_train, batch_size = 1, epochs = 1)

#Create the testing dataset 
#Create a new array containing scaled values from index 1600 to 2075
test_data = scaled_data[training_data_len - 60 : , :]
#Create the data sets x_test and y_test 
x_test = []
y_test = dataset[training_data_len: , :]
for i in range(60, len(test_data)):
	x_test.append(test_data[i-60:i, 0])

#Convert the data to a numpy array
x_test = np.array(x_test) 

#Reshape the data 
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

#Get the model's predicted price values 
predictions = model.predict(x_test)
#Unscaling the values
predictions = scaler.inverse_transform(predictions)

#Evaluate the model by getting the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(((predictions - y_test)**2)))
print(rmse) #7.827112615964164 

#Plot the data 
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions 
#Visualize the model 
plt.figure(figsize=(16, 8))
plt.title('Model')
plt.xlabel('Data', fontsize = 18)
plt.ylabel('Close Price USD ($)', fontsize = 18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Validation', 'Predictions'], loc="lower right")
plt.show()

#Show the valid and predicted prices 
print(valid)

#Get the quote 
apple_quote = web.DataReader('AAPL', data_source='yahoo', start = '2012-01-01', end='2020-04-01')
#Create a new dataframe 
new_df = apple_quote.filter(['Close'])
#Get the last 60 day closing price values and convert the dataframe to an array 
last_60_days = new_df[-60:].values 
#Scale the data to be values between 0 and 1
#use the same min and max values we used at the beginning
last_60_days_scaled = scaler.transform(last_60_days)

#Create an empty list
X_test = []
#Append the past 60 days to this list
X_test.append(last_60_days_scaled)
#Convert the X_test data set to a numpy array 
X_test = np.array(X_test)
#Reshape the data 
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#Get the predicted scaled price
pred_price = model.predict(X_test)
#Undo the scaling
pred_price = scaler.inverse_transform(pred_price)
print(pred_price) #This is what our model predicts for 4/2/2020 

#Get the quote for the actual price on 4/2/2020
apple_quote2 = web.DataReader('AAPL', data_source='yahoo', start = '2020-04-02', end='2020-04-02')
print(apple_quote2['Close'])






