"""
StrategyLearner
"""

import datetime as dt
import QLearner as ql
import pandas as pd
import numpy as np
import util as ut

class StrategyLearner(object):

    # constructor
    def __init__(self, verbose = False):
        self.verbose = verbose
        self.actions = ['EXIT', 'LONG', 'SHORT']
        self.converged_threshold = .5

    def __discretize(self, data, \
        col, \
        symbol, \
        steps=10):

        step_size = len(data) / steps

        # create discretized data dataframe and extract the date index
        discretized_data = pd.DataFrame(data, columns=[symbol])
        discretized_data['Date'] = data.index
        discretized_data.sort([symbol], inplace=True)

        # set the index to numbers to easily slice on
        discretized_data.index = range(0,len(data))
        discretized_data[col] = np.nan

        # put each slice of data into its bucket
        for i in range(0, steps):
            discretized_data.ix[i*step_size:i*step_size+step_size-1, col] = i

        # fill anything we forgot to discretize
        discretized_data.fillna(method='ffill', inplace=True)

        # add the data back as the index
        discretized_data.index = discretized_data['Date']

        # drop the superfluous columns
        discretized_data.drop('Date', axis=1, inplace=True)
        discretized_data.drop(symbol, axis=1, inplace=True)

        # sort back on the day
        discretized_data.sort_index(inplace=True)
        discretized_data[col].astype(int)
        return discretized_data

    def __calculate_discretized_state(self, prices, symbol, prices_SPY):
        rolling_mean = pd.rolling_mean(prices, window=20,center=False)
        rolling_mean.fillna(method='backfill', inplace=True)
        discretized_rollingmean = self.__discretize(rolling_mean[symbol], 'Rolling Mean', symbol)

        momentum = self.__compute_momentum(prices, 5)
        momentum.fillna(method='backfill', inplace=True)
        discretized_momentum = self.__discretize(momentum[symbol], 'Momentum', symbol)

        relative =  prices[symbol] / prices_SPY
        discretized_relative = self.__discretize(relative, 'Relative', symbol)
        return discretized_rollingmean.join([discretized_momentum, discretized_relative])

    def __concat_state(self, values):
        state = ""
        for num in values:
            state += str(int(num))
        return int(state)

    def __get_position(self, action, positions):
        if action == 0:
            return 'EXIT'
        elif action == 1:
            return 'LONG'
        else: 'SHORT'

    def __calculate_reward(self, prev_action, prev_price, curr_price):
        if pd.isnull(prev_action)[0]:
            return 0

        change = (curr_price - prev_price) / prev_price * 100
        if prev_action[0] == 0:
            return 0
        elif prev_action[0] == 500:
            return change
        elif prev_action[0] == -500:
            return -change
        else:
            return 0

    def __converged(self, prev_reward, reward):
        diff = reward - prev_reward
        if -self.converged_threshold < diff < self.converged_threshold:
            return True
        else:
            return False

    def __convert_output(self, trades, prices):
        trades_df = prices.copy()
        trades_df.ix[:, :] = 0
        for index, row in trades.iterrows():
            if row.Order == 'SELL':
                trades_df.loc[row.Date] = -row.Shares
            elif row.Order == 'BUY':
                trades_df.loc[row.Date] = row.Shares
        return trades_df


    def __determine_position(self, prev_position, action):

        if np.isnan(prev_position[0]) and action == 'EXIT':
            return 0, [np.nan, np.nan, np.nan, np.nan]
        elif np.isnan(prev_position[0]) and action == 'LONG':
            return 500, [np.nan, np.nan, 'BUY', 500]
        elif np.isnan(prev_position[0]) and action == 'SHORT':
            return -500, [np.nan, np.nan, 'SELL', 500]

        elif prev_position[0] == -500 and action == 'EXIT':
            return 0, [np.nan, np.nan, 'BUY', 500]
        elif prev_position[0] == -500 and action == 'LONG':
            return 500, [np.nan, np.nan, 'BUY', 1000]
        elif prev_position[0] == -500 and action == 'SHORT':
            return -500, [np.nan, np.nan, np.nan, np.nan]

        elif prev_position[0] == 0 and action == 'EXIT':
            return 0, [np.nan, np.nan, np.nan, np.nan]
        elif prev_position[0] == 0 and action == 'LONG':
            return 500, [np.nan, np.nan, 'BUY', 500]
        elif prev_position[0] == 0 and action == 'SHORT':
            return -500, [np.nan, np.nan, 'SELL', 500]

        elif prev_position[0] == 500 and action == 'EXIT':
            return 0, [np.nan, np.nan, 'SELL', 500]
        elif prev_position[0] == 500 and action == 'LONG':
            return 500, [np.nan, np.nan, np.nan, np.nan]
        elif prev_position[0] == 500 and action == 'SHORT':
            return -500, [np.nan, np.nan, 'SELL', 1000]

    def __compute_momentum(self, prices, n):
        momentum = (prices / prices.shift(n)) - 1
        return pd.DataFrame(momentum)


    def __normalize_prices(self, prices):
        return prices / prices.ix[0]


    # this method should create a QLearner, and train it for trading
    def addEvidence(self, symbol = "IBM", \
        sd=dt.datetime(2008,1,1), \
        ed=dt.datetime(2009,1,1), \
        sv = 10000):

        self.learner = ql.QLearner(num_actions=3, num_states=1000)
        # add your code to do learning here

        # example usage of the old backward compatible util function
        syms=[symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols
        prices_SPY = prices_all['SPY']  # only SPY, for comparison later
        if self.verbose: print(prices)
  
        converged = False
        prev_reward = 0
        while not converged:
            # calculate discretized indicators
            discretized_df = self.__calculate_discretized_state(self.__normalize_prices(prices), symbol, self.__normalize_prices(prices_SPY))

            # discretize state
            discretized_state = self.__concat_state(discretized_df.ix[0].values)
            self.learner.querysetstate(discretized_state)
            positions_df = pd.DataFrame(index=prices.index, columns=['Position'])
            trades_df = pd.DataFrame(index=prices.index, columns=['Date', 'Symbol', 'Order', 'Shares'])
            total_reward = 0
            for index, row in discretized_df.iterrows():
                reward = self.__calculate_reward(positions_df.shift(1).loc[index], \
                                                 prices[symbol].shift(1)[index], \
                                                 prices[symbol].loc[index])
                total_reward += reward
                s_prime = self.__concat_state(row.values)
                action = self.learner.query(s_prime, reward)
                positions_df.loc[index], trades_df.loc[index] = \
                    self.__determine_position(positions_df.shift(1).loc[index], self.actions[action])

            trades_df['Symbol'] = trades_df['Symbol'].apply(lambda x: symbol)
            trades_df['Date'] = prices.index
            trades_df.dropna(inplace=True)
            trades_df.to_csv('trades_training.csv', index=False)
            # check if converged
            converged = self.__converged(prev_reward, total_reward)
            prev_reward = total_reward
        pass


    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol = "IBM", \
        sd=dt.datetime(2009,1,1), \
        ed=dt.datetime(2010,1,1), \
        sv = 10000):

        # here we build a fake set of trades
        # your code should return the same sort of data
        syms=[symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols
        prices_SPY = prices_all['SPY']  # only SPY, for comparison later

        # calculate discretized indicators
        discretized_df = self.__calculate_discretized_state(self.__normalize_prices(prices), symbol, self.__normalize_prices(prices_SPY))
        positions_df = pd.DataFrame(index=prices.index, columns=['Position'])
        trades_df = pd.DataFrame(index=prices.index, columns=['Date', 'Symbol', 'Order', 'Shares'])
        total_reward = 0
        for index, row in discretized_df.iterrows():
            reward = self.__calculate_reward(positions_df.shift(1).loc[index], \
                                             prices[symbol].shift(1)[index], \
                                             prices[symbol].loc[index])
            total_reward += reward
            s_prime = self.__concat_state(row.values)
            action = self.learner.query(s_prime, reward)
            positions_df.loc[index], trades_df.loc[index] = \
                self.__determine_position(positions_df.shift(1).loc[index], self.actions[action])

        trades_df['Symbol'] = trades_df['Symbol'].apply(lambda x: symbol)
        trades_df['Date'] = prices.index
        trades_df.dropna(inplace=True)
        trades_df.to_csv('trades_test.csv', index=False)
        return self.__convert_output(trades_df, prices)

if __name__=="__main__":
    print("One does not simply think up a strategy")
