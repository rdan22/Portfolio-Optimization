'''
Portfolio optimization is the process of selecting the best portfolio,
out of the set of  portfolios being considered, according to some objective. The objective
typically maximizes factors such as expected return, and minimizes costs like financial risk.
'''

#Description: This program attempts to optimize a user's portfolio using the Efficient Frontier

#Import the libraries
from pandas_datareader import data as web
import pandas as pd 
import numpy as np 
from datetime import datetime
import matplotlib.pyplot as plt 
plt.style.use('fivethirtyeight')

#Get the stock symbols/tickers in the portfolio
#FAANG
assets = ['FB', 'AMZN', 'AAPL', 'NFLX', 'GOOG']

#Assign weights to the stocks
weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

#Get the stock/portfolio starting date
stockStartDate = '2013-01-01'

#Get the stocks ending data(today)
today = datetime.today().strftime('%Y-%m-%d')
print(today)

#Create a dataframe to store the adjusted close price of the stocks
df = pd.DataFrame()

#Store the adjusted close price of the stock into the df
for stock in assets:
	df[stock] = web.DataReader(stock, data_source='yahoo', start = stockStartDate, end = today)['Adj Close']

#Show the dataframe
print(df)

#Visually show the stock / portfolio
title = 'Portfolio Adj. Close Price History'
#Get the stocks
my_stocks = df
#Create and plot the graph
for c in my_stocks.columns.values:
	plt.plot(my_stocks[c], label=c)

plt.title(title)
plt.xlabel('Date', fontsize=18)
plt.ylabel('Adj. Price in USD ($)', fontsize = 18)
plt.legend(my_stocks.columns.values, loc='upper left')
plt.show()

#Show the daily simple return
returns = df.pct_change()
print(returns)

#Create and show annualized covariance matrix
#252 is num trading days in a year
cov_matrix_annual = returns.cov() * 252
print(cov_matrix_annual)

#Calculate the portfolio variance
port_variance = np.dot(weights.T, np.dot(cov_matrix_annual, weights))
print(port_variance)

#Calculate the portfolio volatility aka standard deviation
port_volatility = np.sqrt(port_variance)
print(port_volatility)

#Calculate the annual portfolio return
portfolioSimpleAnnualReturn = np.sum(returns.mean() * weights) * 252
print(portfolioSimpleAnnualReturn)

#Show the expected annual return, volatility (risk), and variance
percent_var = str(round(port_variance, 2) * 100) + '%'
percent_vols = str(round(port_volatility, 2) * 100) + '%'
percent_ret = str(round(portfolioSimpleAnnualReturn, 2) * 100) + '%'

print('Expected annual return: ' + percent_ret)
print('Annual volatility /risk: ' + percent_vols)
print('Annual variance: ' + percent_var)

from pypfopt.efficient_frontier import EfficientFrontier 
from pypfopt import risk_models
from pypfopt import expected_returns

#Portfolio Optimization 
#Calculate the expected returns and the annualized sample covariance matrix of asset returns
mu = expected_returns.mean_historical_return(df)
S = risk_models.sample_cov(df)

#Optimize for max sharpe ratio (a way to describe how much excess return you receive for some amount of volatility)
#It measures the performance of an investment compared to an investment that is risk-free (treasury bonds, etc)
ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe()
#helper method to clean raw weights and sets any weights whose abs value below some cutoff to 0 and rounds the rest of the values 
cleaned_weights = ef.clean_weights()
print(cleaned_weights)
print(ef.portfolio_performance(verbose = True))

#Get the discrete allocation of each share per stock
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

latest_prices = get_latest_prices(df)
weights = cleaned_weights
da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=15000)

allocation, leftover = da.lp_portfolio()
print('Discrete Allocation: ', allocation)
print('Funds Remaining: ${:.2f}'.format(leftover))

























