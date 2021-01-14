"""Stock Portfolio.ipynb

Portfolio optimization is the process of selecting the best portfolio, 
out of the set of portfolios being considered, according to some objective. 
The objective typically maximizes factors such as expected return and 
minimizes costs like financial risk.
"""

#Description: This program optimizes a stock portfolio.

#Import the libraries
import pandas as pd
import pandas_datareader as web
import numpy as np
import requests

tickers = pd.read_csv('NYSE.csv')['Symbol']
df = pd.DataFrame()
for symbol in tickers:
  df[symbol] = web.DataReader(symbol, data_source = 'yahoo', start = '2016-01-06', end = '2021-01-06')['Adj Close']

df

#Get the assets/tickers
assets = df.columns

pip install PyPortfolioOpt

#Optimize the portfolio
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

#Calculate the expected annualized returns and the annualized sample covariance matrix of the daily asset returns
mu = expected_returns.mean_historical_return(df)
S = risk_models.sample_cov(df)

#Optimize for the maximal Sharpe ratio
ef = EfficientFrontier(mu, S) #create the efficient frontier object
weights = ef.max_sharpe()

cleaned_weights = ef.clean_weights()
print(cleaned_weights)
ef.portfolio_performance(verbose=True)

pip install pulp

#Get the discrete allocation of each share per stock
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

portfolio_val = 5000
latest_prices = get_latest_prices(df)
weights = cleaned_weights
da = DiscreteAllocation(weights, latest_prices, total_portfolio_value = portfolio_val)
allocation, leftover = da.lp_portfolio()
print('Discrete allocation:', allocation)
print('Remaining funds: $', leftover)

#Create a function to get the companies names
def get_company_name(symbol):
  url = 'http://d.yimg.com/autoc.finance.yahoo.com/autoc?query='+symbol+'&region=1&lang=en'
  result = requests.get(url).json()
  for r in result['ResultSet']['Result']:
    if r['symbol'] == symbol:
      return r['name']

#Store the company name into a list
company_name = []
for symbol in allocation:
  company_name.append(get_company_name(symbol))

#Get the discrete allocation values
discrete_allocations = []
for symbol in allocation:
  discrete_allocations.append(allocation.get(symbol))

#Create a dataframe for the portfolio
portfolio_df = pd.DataFrame(columns= ['Company_name', 'Company_ticker', 'Discrete_val_'+str(portfolio_val)])

portfolio_df['Company_name'] = company_name
portfolio_df['Company_ticker'] = allocation
portfolio_df['Discrete_val_'+str(portfolio_val)] = discrete_allocations

#Show the portfolio
portfolio_df