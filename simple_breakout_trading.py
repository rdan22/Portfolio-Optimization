#Implemented through QuantConnect's Algorithm Lab

import numpy as np # To calculate the std of volatility adjusted lookback length

class SimpleBreakoutExample(QCAlgorithm):

    def Initialize(self):
        # Set the cash amount for backtest only - in real life, it's taken from broker's account
        self.SetCash(100000)
        
        # Start and end dates for backtest - doesn't matter right now but leave enough data untouched for testing
        self.SetStartDate(2017,1,1)
        self.SetEndDate(2021,1,1)
        
        # Add asset - the security used for trading
        self.symbol = self.AddEquity("SPY", Resolution.Daily).Symbol
        
        # Lookback length for breakout point (in days)
        # Our algorithm will dynamically change this lookback length based on changes in volatility
        self.lookback = 20
        
        # Upper/lower limit for lookback length
        self.ceiling, self.floor = 30, 10
        
        # Price offset for stop order
        self.initialStopRisk = 0.98 # How close our first stop loss is to the security's price - allow a 2% loss
        self.trailingStopRisk = 0.9 # How close our trading stop will follow the asset's price - trade at the price by 10%
        
        # Schedule function 20 minutes after every market open
        self.Schedule.On(self.DateRules.EveryDay(self.symbol), \
                        self.TimeRules.AfterMarketOpen(self.symbol, 20), \
                        Action(self.EveryMarketOpen))


    def OnData(self, data):
        # Plot security's price and compare our algorithm's performance
        self.Plot("Data Chart", self.symbol, self.Securities[self.symbol].Close)

 
    def EveryMarketOpen(self):
        # Dynamically determine lookback length based on 30 day volatility change rate
        # Compare it with the same value from yesterday
        close = self.History(self.symbol, 31, Resolution.Daily)["close"] # Close price of the security over the past 31 days
        todayvol = np.std(close[1:31]) # Take the standard deviation
        yesterdayvol = np.std(close[0:30]) # Do it for the day before
        deltavol = (todayvol - yesterdayvol) / todayvol # Normalized difference between today's and yesterday's std
        self.lookback = round(self.lookback * (1 + deltavol)) # Lookback = lookback * (1 + deltavol) to ensure it increases when volatility increases
        
        # Make sure it is within upper/lower limit of lockback length, inclusive
        if self.lookback > self.ceiling:
            self.lookback = self.ceiling
        elif self.lookback < self.floor:
            self.lookback = self.floor
        
        # List of daily highs
        self.high = self.History(self.symbol, self.lookback, Resolution.Daily)["high"]
        
        # Buy SPY at the market price in case of breakout
        # Leave out the last data point from high since we don't want to compare yesterday's high and low
        if not self.Securities[self.symbol].Invested and \
                self.Securities[self.symbol].Close >= max(self.high[:-1]):
            self.SetHoldings(self.symbol, 1) # Security and percentage of our portfolio allocated to this position
            self.breakoutlvl = max(self.high[:-1]) # Save the breakout level
            self.highestPrice = self.breakoutlvl # Set highest price to this breakout level
        
        
        # Create trailing stop loss if invested 
        if self.Securities[self.symbol].Invested:
            # If no open orders exist, send stop-loss
            if not self.Transactions.GetOpenOrders(self.symbol):
                # Send a stop market order: 
                # The security, number of the shares (- symbolizes sell order), stop-loss price = breakout level * initial stop risk 
                self.stopMarketTicket = self.StopMarketOrder(self.symbol, \
                                        -self.Portfolio[self.symbol].Quantity, \
                                        self.initialStopRisk * self.breakoutlvl)
            
            # Check if the asset's price is higher than highestPrice and trailing stop price not below initial stop price
            if self.Securities[self.symbol].Close > self.highestPrice and \
                    self.initialStopRisk * self.breakoutlvl < self.Securities[self.symbol].Close * self.trailingStopRisk:
                # Save the new high (latest close price) to highestPrice 
                self.highestPrice = self.Securities[self.symbol].Close
                # Update the stop price 
                updateFields = UpdateOrderFields()
                # Update the order price of our stop loss so that it rises with the security's price
                # Stop price = latest close price * trailing stop risk
                updateFields.StopPrice = self.Securities[self.symbol].Close * self.trailingStopRisk
                # Update existing stop-loss order with the update method
                self.stopMarketTicket.Update(updateFields)
                
                # Print the new stop price with Debug()
                self.Debug(updateFields.StopPrice)
            
            # Plot trailing stop's price
            self.Plot("Data Chart", "Stop Price", self.stopMarketTicket.Get(OrderField.StopPrice))