#We use Facebook's NeuralProphet to build more sophisticated
#deep learning models for time-series prediction.

import pandas as pd
from neuralprophet import NeuralProphet

import pandas_datareader as pdr 
from datetime import datetime
import matplotlib.pyplot as plt 

#Consider the daily stock price data for the S&P 500 Index for the last 10 years. 

start = datetime(2010, 12, 13)
end = datetime(2020, 12, 18)

sp500_data = pdr.get_data_fred('sp500', start, end)
plt.figure(figsize=(10, 7))
plt.plot(sp500_data)
plt.title('S&P 500 Prices')
plt.show()

#In order to train NeuralProphet on a dataset, we need to make
#sure the data is formatted so that the date column is named
#ds and the column with the target variable is named y.  

sp500_data = sp500_data.reset_index().rename(columns={'DATE': 'ds', 'sp500':'y'}) 

#With NeuralProphet, we can model trends in time-series data by specifying a few args.
# - n_changepoints - specifies the number of points where the broader trend (rate of
#	increase/decrease) in the data changes. 
# - trend_reg - a regularization parameter that controls the flexibility of changepoint
# 	selection. Larger values (~1-100) will limit the variability of changepoints. Smaller
#	values (~0.001-1.0) will allow for more variability in changepoints. 

model = NeuralProphet(n_changepoints=100, trend_reg=0.05, yearly_seasonality=False, 
		weekly_seasonality=False, daily_seasonality=False)

#The fit function uses the following parameters:
# - validate_each_epoch - a flag indicating whether or not to 
#	validate the model's performance on the validation data 
#	in each epoch.
# - valid_p - a float between 0 and 1 indicating the proportion
#	of data that should be used for validation. 
# - plot_live_loss - a flag indicating whether or not to generate
# 	a live plot of the model's training and validation loss.
# - epochs - the number of epichs that the models should be trained for.

metrics = model.fit(sp500_data, validate_each_epoch=True, valid_p=0.2, freq='D', plot_live_loss=True, epochs=100)

def plot_forecast(model, data, periods, historic_pred=True, highlight_steps_ahead=None):
	""" plot_forecast function - generates and plots the forecast for a NeuralProphet model
		- model -> a trained NeuralProphet model
		- data -> the dataframe used for training
		- periods -> the number of periods to forecast
		- historic_pred -> a flag indicating whether or not to plot the model's predictions 
		  on historic data
		- highlight_steps_ahead -> the number of steps ahead of the forecast line to highlight, 
		  used for autoregressive models only
	"""	
	future = model.make_future_dataframe(data, periods=periods, n_historic_predictions=historic_pred)
	forecast = model.predict(future)
	if highlight_steps_ahead is not None:
		model = model.highlight_nth_step_ahead_of_each_forecast(highlight_steps_ahead)
		model.plot_last_forecast(forecast)
		plt.show()
	else:
		model.plot(forecast)
		plt.show()

#Using the above function we can visualize the model's S&P 500 price predictions on 
#historical data and its forecast for the next 60 days as shown below:

plot_forecast(model, sp500_data, periods=60)

#The model seems to suffer from underfitting, particularly when we look at the historical
#data from Jan 2019 to December 2020 that was likely used for validation. We can take a look at
#just the model's forecast without the predictions on the historical data to see what's happening. 
#This naive forecast follows a straight line, but we can make it more realistic by adding
#seasonality to it. 

model = NeuralProphet(n_changepoints=100, trend_reg=0.05, yearly_seasonality=True, 
		weekly_seasonality=False, daily_seasonality=False)
metrics = model.fit(sp500_data, validate_each_epoch=True, valid_p=0.2, freq='D', plot_live_loss=True, epochs=100)
plot_forecast(model, sp500_data, periods=60)

#By adding yearly seasonality, we see that the naive plot now shows a smooth curve but it 
#still suffers from underfitting. We can capture the volatility by using an autoregressive
#model such as an AR-Net. 

#AR-Net is an autoregressive neural network used for time-series forecasting. Autoregressive
#models use past historical data from previous time steps to generate predictions for the next
#time steps. The values of the target variable in the previous time steps are parameters that 
#serve as inputs for the models. 

#We can train a model that uses the price of the S&P 500 data from the past 60 days to predict 
#the price for the next 60 days. These parameters are specified by the n_lags and n_forecasts args
#in the code below:

model = NeuralProphet(n_forecasts=60, n_lags=60, n_changepoints=100, 
	yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
	batch_size=64, epochs=100, learning_rate=1.0)
model.fit(sp500_data, freq='D', valid_p=0.2, epochs=100)

#Plotting the forecast for the AR-Net model demonstrates how much better it really is when
#it comes to capturing movements in the stock market:

plot_forecast(model, sp500_data, periods=60, historic_pred=True)

#The AR-Net model generates more realistic predictions and manages to capture some jagged lines
#in the movements of the stock market. 
