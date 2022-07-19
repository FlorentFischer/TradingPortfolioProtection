#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import time





option_chain = pd.read_excel('/home/TradingPortfolioProtecion/mysite/option_chain.xls', index_col = 'Date')
Dataset = pd.read_excel('/home/TradingPortfolioProtecion/mysite/Dataset.xls', index_col = 'Date')




# In sample dataframe
in_sample = (Dataset.index > '1960-01-01') & (Dataset.index <= '2000-01-01')
df_in = Dataset.loc[in_sample]


# In[6]:


# Out-sample dataframe:
out_sample = (Dataset.index > '1998-01-01') & (Dataset.index <= '2022-01-01')
df_out = Dataset.loc[out_sample].dropna()
df_out = pd.concat([df_out,option_chain], axis = 1).dropna()


# ML-DL Stretgies
svr_strategy = pd.read_excel('/home/TradingPortfolioProtecion/mysite/svr_strategy.xls', index_col = 'Date')
ann_strategy = pd.read_excel('/home/TradingPortfolioProtecion/mysite/ann_strategy.xls', index_col = 'Date')
rnn_strategy = pd.read_excel('/home/TradingPortfolioProtecion/mysite/rnn_strategy.xls', index_col = 'Date')

# Heatmaps
perf_emv = pd.read_excel('/home/TradingPortfolioProtecion/mysite/perf_emv.xls', index_col = 'Unnamed: 0')
perf_rsi = pd.read_excel('/home/TradingPortfolioProtecion/mysite/perf_rsi.xls', index_col = 'Unnamed: 0')
perf_macd = pd.read_excel('/home/TradingPortfolioProtecion/mysite/perf_macd.xls', index_col = 'Unnamed: 0')

# Studies
study_ann = pd.read_excel('/home/TradingPortfolioProtecion/mysite/study_ann.xls', index_col = 'Unnamed: 0')
study_svr = pd.read_excel('/home/TradingPortfolioProtecion/mysite/study_svr.xls', index_col = 'Unnamed: 0')
study_rnn = pd.read_excel('/home/TradingPortfolioProtecion/mysite/study_rnn.xls', index_col = 'Unnamed: 0')


class Rebalancing:
    """Class"""

    def __init__(self, database,svr,ann,rnn):
        # INPUTS
        self.database = database
        self.svr = svr
        self.ann = ann
        self.rnn = rnn

        # VARIABLES
        self.portfolio = None



    # MACD indicator function

    def MACD(self,parameter_1,paramter_2):

        '''
        parameter_1: corresponds to the small window for EMA computation
        parameter_1: corresponds to the large window for EMA computation
        '''
        macd = self.database.copy()
        self.strategy = 'macd'

        #_____________________________________________________________________________
        #          MACD formula implementation
        #_____________________________________________________________________________

        # MACD construction
        macd['EWMA_small'] = macd['Adj Close'].ewm(parameter_1).mean().shift(1).dropna()
        macd['EWMA_large'] = macd['Adj Close'].ewm(paramter_2).mean().shift(1).dropna()
        macd['MACD'] = macd['EWMA_small']- macd['EWMA_large']

        # Adjusted MACD
        macd['vol_price'] = macd['Adj Close'].rolling(63).std().shift(1)
        macd['q'] = macd['MACD'] / macd['vol_price']
        macd['q_vol'] = macd['q'].rolling(252).std()
        macd['Indicator'] = macd['q']/(macd['q_vol'])

        # Signal creation
        macd['position_sizing'] = (macd['Indicator'] * np.exp((-macd['Indicator']**2)/4))/0.89
        macd['Signal'] = np.nan
        macd['Signal'].loc[macd['position_sizing'] < 0] = 0
        macd['Signal'].loc[macd['position_sizing'] > 0] = 1


        #_____________________________________________________________________________
        #          Inclusion of the cost
        #_____________________________________________________________________________

        # Signal and cost list
        signal_list = np.array(macd['Signal']).tolist()
        cost = []

        # Cost inclusion
        for i in range(1,len(macd['Signal'])):
            if (signal_list[i] == 0) and (signal_list[i-1] == 1):
                cost.append(-0.005)
            else:
                cost.append(0)
        cost.append(0)
        macd['Cost'] = pd.DataFrame(np.array(cost), index = macd.index)


        #_____________________________________________________________________________
        #          Strategy's results
        #_____________________________________________________________________________

        # Strategy
        macd['Strategy'] = (macd['Signal'] * macd['Change']) + macd['Cost']
        macd_result = pd.concat((macd['Indicator'],macd['Signal'],macd['Strategy'],macd['Change'],macd['60/40']), axis  = 1)
        self.portfolio = macd_result.copy().dropna()




    # RSI indicator function
    def RSI(self,parameter):

        '''
        parameter: corresponds to the rolling window for EMA computation
        result: is a boolean allowing to return the dataframe of the strategy creation
        '''
        rsi = self.database.copy()
        self.strategy = 'rsi'

        #_____________________________________________________________________________
        #          RSI formula implementation
        #_____________________________________________________________________________

        # RSI indicator construction
        rsi['Diff'] = rsi['Adj Close'].diff()
        rsi['Up'], rsi['Down'] = np.zeros(len(rsi)), np.zeros(len(rsi))
        rsi['Up'].loc[rsi['Diff'] >= 0] = rsi['Diff']
        rsi['Down'].loc[rsi['Diff'] < 0] = rsi['Diff']
        rsi['U'] = rsi['Up'].ewm(parameter).mean().shift(1).dropna()
        rsi['D'] = abs(rsi['Down'].ewm(parameter).mean().shift(1).dropna())
        rsi['Indicator'] = 100 - 100/(1+(rsi['U'] / rsi['D']))

        # Signal creation
        rsi['Signal'] = np.zeros(len(rsi))
        rsi['Signal'].loc[(rsi['Indicator'] > 50) & (rsi['Indicator'] < 70)] = 1

        a,b = 30,70
        list_signal = np.array(rsi['Signal']).tolist()
        list_RSI  = np.array(rsi['Indicator']).tolist()

        for h in range(len(rsi['Indicator'])):
            # Buy after oversell
            if list_RSI[h] <= a:
              list_signal[h] = 1
              a,b = 50,70
            else:
              a = 30
            # Block buy after overbought)
            if list_RSI[h] >= b:
              list_signal[h] = 0
              b,a = 50,30
            else:
              b = 70


        #_____________________________________________________________________________
        #          Inclusion of the cost
        #_____________________________________________________________________________

        # Signal and cost list
        rsi['Signal'] = pd.DataFrame(np.array(list_signal), index = rsi.index)
        signal_list = np.array(rsi['Signal']).tolist()
        cost = []

        # Cost inclusion
        for i in range(1,len(rsi)):
            if (signal_list[i] == 0) and (signal_list[i-1] == 1):
              cost.append(-0.005)
            else:
              cost.append(0)
        cost.append(0)
        rsi['Cost'] = pd.DataFrame(np.array(cost), index = rsi.index)


        #_____________________________________________________________________________
        #          Strategy's results
        #_____________________________________________________________________________

        # Strategy
        rsi['Strategy'] = rsi['Signal'] * rsi ['Change'] + rsi['Cost']
        rsi_result = pd.concat((rsi['Indicator'],rsi['Signal'],rsi['Strategy'],rsi['Change'],rsi['60/40']), axis  = 1)
        self.portfolio = rsi_result.copy().dropna()



    def EVM(self,parameter):

        '''
        parameter: corresponds to the rolling window for EMA computation
        '''

        emv = self.database.copy()
        self.strategy = 'emv'

        #_____________________________________________________________________________
        #          EMV formula implementation
        #_____________________________________________________________________________

        # EMV construction
        distance_moved = ((emv['High'].shift(1) + emv['Low'].shift(1))/2) - ((emv['High'].shift(2) + emv['Low'].shift(2))/2)
        box_ratio = (emv['Volume'].shift(1) / 10000000) / ((emv['High'].shift(1) - emv['Low'].shift(1)))
        EVM = distance_moved / box_ratio
        emv['Indicator'] = EVM.rolling(parameter).mean()

        # Signal creation
        emv['Signal'] = np.nan
        emv['Signal'].loc[emv['Indicator'] <= -1] = 0
        emv['Signal'].loc[emv['Indicator'] > -1] = 1


        #_____________________________________________________________________________
        #          Inclusion of the cost
        #_____________________________________________________________________________

        # Signal and cost list
        signal_list = np.array(emv['Signal']).tolist()
        cost = []

        # Cost inclusion
        for i in range(1,len(emv)):
            if (signal_list[i] == 0) and (signal_list[i-1] == 1):
                cost.append(-0.005)
            else:
                cost.append(0)
        cost.append(0)
        emv['Cost'] = pd.DataFrame(np.array(cost), index = emv.index)


        #_____________________________________________________________________________
        #          Strategy's results
        #_____________________________________________________________________________

        # Strategy
        emv['Strategy'] = (emv['Signal'] * emv['Change']) + emv['Cost']
        emv_result = pd.concat((emv['Indicator'],emv['Signal'],emv['Strategy'],emv['Change'],emv['60/40']), axis  = 1)
        self.portfolio = emv_result.copy().dropna()

    def SVR(self):

        data = self.database.copy()
        svr = self.svr.copy()

        self.strategy = 'svr'
        svr_strategy = pd.concat([svr,data['Change'],data['60/40']], axis = 1).dropna()
        self.portfolio = svr_strategy

    def ANN(self):

        data = self.database.copy()
        ann = self.ann.copy()

        self.strategy = 'ann'
        ann_strategy = pd.concat([ann,data['Change'],data['60/40']], axis = 1).dropna()
        self.portfolio = ann_strategy

    def RNN(self):

        data = self.database.copy()
        rnn = self.rnn.copy()

        self.strategy = 'rnn'
        rnn_strategy = pd.concat([rnn,data['Change'],data['60/40']], axis = 1).dropna()
        self.portfolio = rnn_strategy


#trade = Rebalancing(df_out,svr_strategy,rnn_strategy)
#trade.RNN()
#trade.portfolio


# In[7]:


trade = Rebalancing(df_out,svr_strategy,ann_strategy,rnn_strategy)
trade.RSI(23)
trade.portfolio


# In[8]:


#p = Protection(df_out,svr_strategy,rnn_strategy,'rsi')
#p.OBPI(0.45)
#vis = Visualization(p.portfolio,'macd')
#vis.performance_protection()


# In[ ]:





# In[9]:


class Protection:

    def __init__(self,database, svr,ann,rnn, strategy):

        self.database = database
        self.svr = svr
        self.rnn = rnn
        self.ann = ann

        # MACD:
        trade = Rebalancing(self.database,self.svr,self.ann,self.rnn)
        trade.MACD(58,76)
        self.macd = trade.portfolio

        # RSI
        trade = Rebalancing(self.database,self.svr,self.ann,self.rnn)
        trade.RSI(67)
        self.rsi = trade.portfolio

        # EMV
        trade = Rebalancing(self.database,self.svr,self.ann,self.rnn)
        trade.EVM(31)
        self.emv = trade.portfolio


        if strategy == 'macd':
            self.portfolio = self.macd.drop(['Change','60/40'],axis = 1)
        if strategy == 'rsi':
            self.portfolio = self.rsi.drop(['Change','60/40'],axis = 1)
        if strategy == 'emv':
            self.portfolio = self.emv.drop(['Change','60/40'],axis = 1)
        if strategy == 'svr':
            self.portfolio = self.svr
        if strategy == 'ann':
            self.portfolio = self.ann
        if strategy == 'rnn':
            self.portfolio = self.rnn

    def OBPI(self,target):

        bdd = self.database
        pf = self.portfolio
        data = pd.concat([bdd,pf],axis = 1).dropna()


        # Individual variables
        rf = 0.01
        vol_target = target
        risky_asset = [1]
        riskless_asset = [0]
        obpi = [0]

        # Prepare the Dataset

        # Option chain cleaning
        Strike_str = ['-5%','-4%','-3%','-2%','-1%','0%','1%','2%','3%','4%','5%']
        Option_Chain = data[Strike_str]


        #Prepare the empty dataframe columns
        data['W_r'] = np.nan
        data['W_s'] = np.nan
        data['OBPI'] = np.nan
        Strike = list(np.linspace(-0.05,0.05,11)+1)


        # Running OBPI with a protective put implementation
        for i in range(1,len(data)):

            # Risk Budgeting on the overall portfolio
            risky_asset.append(min(vol_target / (data['VIX'].iloc[i-1]/100), 1))
            riskless_asset.append(1 - risky_asset[i])

            # Find optimal strike given risk budgeting level
            position = int(min(riskless_asset[i]*10,10))


            # OBPI strategy:
            X = (data['Adj Close'].iloc[i-1]*Strike[position])
            ST = (data['Adj Close'].iloc[i])
            maximum = np.max((ST,X))
            put = data[Strike_str[position]].iloc[i-1] * np.exp(rf*2/252)
            obpi.append(((maximum - data['Adj Close'].iloc[i]) / data['Adj Close'].iloc[i-1])  - (put / data['Adj Close'].iloc[i]))

        # Add Risk budgeting weights
        data['W_r'] = np.array(risky_asset)
        data['W_s'] = np.array(riskless_asset)



        # Return OBPI without trading strategy combination
        data['OBPI'] =  data['Change'] * data['W_r'] + np.array(obpi)*data['W_s']



        # Signal and cost list
        signal_list = np.array(data['Signal']).tolist()
        cost = []

        # Cost inclusion
        for i in range(1,len(data)):
            if ((signal_list[i] == 0) and (signal_list[i-1] == 1)) or ((signal_list[i] == 1) and (signal_list[i-1] == 0)):
              cost.append(-0.005)
            else:
              cost.append(0)
        cost.append(0)

        data['Cost'] = pd.DataFrame(np.array(cost), index = data.index)

        # Return OBPI with trading strategy combination
        OBPI_strategy = pd.DataFrame(data['Signal'] * data['OBPI'] + np.where(data['Signal'] == 0,1,0)*data['US Bond'] + data['Cost'],columns =["Strategy_protection"])
        protection = pd.concat([OBPI_strategy,data['W_r'],data['W_s'],data['60/40'],data['Signal'],self.portfolio['Strategy']], axis = 1)
        self.portfolio = protection.dropna()


    def CPPI(self,multiplier, rebalancing):
        # Strategy parameters
        n = 252
        protected = 1
        m = multiplier
        rate = 0.03
        k = 1

        bdd = self.database
        pf = self.portfolio
        data = pd.concat([bdd,pf],axis = 1).dropna()

        # Dataframe treatment
        returns = np.array(data['Change']).tolist()
        bond = np.array(data['US Bond']).tolist()
        stock = k * ((1+np.array(returns)).cumprod()).tolist()

        # Shift of data
        stock.insert(0,1)
        returns.insert(0,0)
        bond.insert(0,0)


        # Initial CPPI parameters
        cppi = k
        f = k*protected*np.exp(-rate*(len(returns)/n))
        c = cppi - f
        r = np.min([m*c,cppi])
        s = cppi - r

        # List
        portfolio = [k]
        floor = [f]
        cushion = [c]
        risky = [r]
        safe = [s]
        w_risk = [r/k]
        w_safe = [s/k]


        # Running the algorithm
        for i in range(1,len(returns)):

            # Portfolio value
            cppi = r * (1+returns[i]) + s * (1+bond[i])
            f = protected * np.exp(-rate*(len(returns)-i)/n)
            c = cppi - f
            r = max(min(m*c,cppi),0)
            s = cppi - r

            # Update lists
            portfolio.append(cppi)
            floor.append(f)
            cushion.append(c)
            risky.append(r)
            safe.append(s)
            w_risk.append(r/cppi)
            w_safe.append(s/cppi)


        # Create a dataframes of all strategy's components
        total = [stock,np.round(portfolio,3),
                np.round(floor,3),np.round(cushion,3),
                np.round(risky,3),np.round(safe,3),
              np.round(w_risk,3),np.round(w_safe,3)]


        weights_cppi = pd.DataFrame(total, index= ['Stock Price', 'Portfolio','Floor',
                                  'Cushion', 'Risky Asset','Safe Asset','W_r','W_s']).transpose()
        weights_cppi = weights_cppi.iloc[:len(weights_cppi)-1,:]
        weights_cppi.index = data.index

        # Rebalancing Interval
        Test = weights_cppi[['W_r', 'W_s']]
        Test = Test.reset_index()
        Test.set_index('Date', inplace=True)
        Test = Test.resample(str(rebalancing)+'M').mean()

        # Add rebalnacing interval to the dataframe
        dataset = pd.concat([data, Test], axis = 1)

        # Dataframe preparation on the trading portfolio
        dataset['W_r'].iloc[0:1] = weights_cppi['W_r'].iloc[0:1]
        dataset['W_s'].iloc[0:1] = weights_cppi['W_s'].iloc[0:1]
        dataset = dataset.fillna(method='ffill')
        dataset['Signal_Safe'] = np.where(dataset['Signal'] == 0,1,0)

        # Signal and cost list
        signal_list = np.array(dataset['Signal']).tolist()
        rebalancing_list = np.array(dataset['W_r']).tolist()
        cost = []

        # Cost inclusion in the strategy
        for i in range(1,len(dataset)):
            if ((signal_list[i] == 0) and (signal_list[i-1] == 1))             or ((signal_list[i] == 1) and (signal_list[i-1] == 0))             or ((rebalancing_list[i]!=rebalancing_list[i-1])):

              cost.append(-0.005)
            else:
              cost.append(0)
        cost.append(0)
        dataset['Cost'] = pd.DataFrame(np.array(cost), index = dataset.index)

        # Final results
        dataset['CPPI_Risky'] = dataset['W_r'] * dataset['Change'] * dataset['Signal']
        dataset['CPPI_Safe'] = dataset['W_s'] * dataset['US Bond'] * dataset['Signal']
        dataset["Strategy_protection"] = dataset['CPPI_Risky'] + dataset['CPPI_Safe'] + dataset['US Bond'] * dataset['Signal_Safe'] + dataset['Cost']
        protection = pd.concat([dataset['Strategy_protection'],dataset['W_r'],dataset['W_s'],dataset['60/40'],dataset['Signal'],self.portfolio['Strategy']], axis = 1)
        self.portfolio = protection.dropna()

    def TIPP(self,multiplier, rebalancing):
        # Strategy parameters
        n = 252
        protected = 1
        m = multiplier
        rate = 0.03
        k = 1

        # Dataframe treatment
        bdd = self.database
        pf = self.portfolio
        data = pd.concat([bdd,pf],axis = 1).dropna()
        returns = np.array(data['Change']).tolist()
        bond = np.array(data['US Bond']).tolist()
        stock = k * ((1+np.array(returns)).cumprod()).tolist()

        # Shift of data
        stock.insert(0,1)
        returns.insert(0,0)
        bond.insert(0,0)


        # Initial CPPI parameters
        cppi = k
        rc = k
        f = k*protected*np.exp(-rate*(len(returns)/n))
        c = cppi - f
        r = np.min([m*c,cppi])
        s = cppi - r

        # List
        portfolio = [k]
        ratchet_capital = [k]
        floor = [f]
        cushion = [c]
        risky = [r]
        safe = [s]
        w_risk = [r/k]
        w_safe = [s/k]


        # Running the algorithm
        for i in range(1,len(returns)):

            # Portfolio value
            cppi = r * (1+returns[i]) + s * (1+bond[i])
            if cppi > rc:
              rc = cppi
              f = rc * protected * np.exp(-rate*(len(returns)-i)/n)
            else:
              rc = rc
              f = f

            c = cppi - f
            r = max(min(m*c,cppi),0)
            s = cppi - r

            # Update lists
            portfolio.append(cppi)
            ratchet_capital.append(rc)
            floor.append(f)
            cushion.append(c)
            risky.append(r)
            safe.append(s)
            w_risk.append(r/cppi)
            w_safe.append(s/cppi)

        # Create a dataframes of all strategy's components
        total = [stock,np.round(portfolio,3),
                np.round(floor,3),np.round(cushion,3),
                np.round(risky,3),np.round(safe,3),
              np.round(w_risk,3),np.round(w_safe,3)]


        weights_cppi = pd.DataFrame(total, index= ['Stock Price', 'Portfolio','Floor',
                                  'Cushion', 'Risky Asset','Safe Asset','W_r','W_s']).transpose()
        weights_cppi = weights_cppi.iloc[:len(weights_cppi)-1,:]
        weights_cppi.index = data.index

        # Rebalancing Interval
        Test = weights_cppi[['W_r', 'W_s']]
        Test = Test.reset_index()
        Test.set_index('Date', inplace=True)
        Test = Test.resample(str(rebalancing)+'M').mean()

        # Add rebalnacing interval to the dataframe
        dataset = pd.concat([data, Test], axis = 1)

        # Dataframe preparation on the trading portfolio
        dataset['W_r'].iloc[0:1] = weights_cppi['W_r'].iloc[0:1]
        dataset['W_s'].iloc[0:1] = weights_cppi['W_s'].iloc[0:1]
        dataset = dataset.fillna(method='ffill')
        dataset['Signal_Safe'] = np.where(dataset['Signal'] == 0,1,0)


        # Signal and cost list
        signal_list = np.array(dataset['Signal']).tolist()
        rebalancing_list = np.array(dataset['W_r']).tolist()
        cost = []

        # Cost inclusion in the strategy
        for i in range(1,len(dataset)):
            if ((signal_list[i] == 0) and (signal_list[i-1] == 1))             or ((signal_list[i] == 1) and (signal_list[i-1] == 0))             or ((rebalancing_list[i]!=rebalancing_list[i-1])):

                cost.append(-0.005)
            else:
                cost.append(0)
        cost.append(0)
        dataset['Cost'] = pd.DataFrame(np.array(cost), index = dataset.index)

        # Final reulsts
        dataset['TIPP_Risky'] = dataset['W_r'] * dataset['Change'] * dataset['Signal']
        dataset['TIPP_Safe'] = dataset['W_s'] * dataset['US Bond'] * dataset['Signal']
        dataset['Strategy_protection'] = dataset['TIPP_Risky'] + dataset['TIPP_Safe'] + dataset['US Bond'] * dataset['Signal_Safe'] + dataset['Cost']
        protection = pd.concat([dataset['Strategy_protection'],dataset['W_r'],dataset['W_s'],dataset['60/40'],dataset['Signal'],self.portfolio['Strategy']], axis = 1)
        self.portfolio = protection.dropna()

    def HOC(self, target, multiplier, min_exp):

        bdd = self.database
        pf = self.portfolio
        data = pd.concat([bdd,pf],axis = 1).dropna()


        # Individual variables
        rf = 0.01
        vol_target = target
        risky_asset = [1]
        riskless_asset = [0]
        obpi = [0]

        # Prepare the Dataset

        # Option chain cleaning
        Strike_str = ['-5%','-4%','-3%','-2%','-1%','0%','1%','2%','3%','4%','5%']
        Option_Chain = data[Strike_str]


        #Prepare the empty dataframe columns
        data['R_w'] = np.nan
        data['RL_w'] = np.nan
        data['OBPI'] = np.nan
        Strike = list(np.linspace(-0.05,0.05,11)+1)


        # Running OBPI with a protective put implementation
        for i in range(1,len(data)):

            # Risk Budgeting on the overall portfolio
            risky_asset.append(min(vol_target / (data['VIX'].iloc[i-1]/100), 1))
            riskless_asset.append(1 - risky_asset[i])

            # Find optimal strike given risk budgeting level
            position = int(min(riskless_asset[i]*10,10))


            # OBPI strategy:
            X = (data['Adj Close'].iloc[i-1]*Strike[position])
            ST = (data['Adj Close'].iloc[i])
            maximum = np.max((ST,X))
            put = data[Strike_str[position]].iloc[i-1] * np.exp(rf*2/252)
            obpi.append(((maximum - data['Adj Close'].iloc[i]) / data['Adj Close'].iloc[i-1])  - (put / data['Adj Close'].iloc[i]))

        # Add Risk budgeting weights
        data['R_w'] = np.array(risky_asset)
        data['RL_w'] = np.array(riskless_asset)



        # Return OBPI without trading strategy combination
        data['OBPI'] =  data['Change'] * data['R_w'] + np.array(obpi)*data['RL_w']


        # Strategy parameters
        n = 252
        protected = 1
        m = multiplier
        mineq = min_exp
        rate = 0.03
        k = 1

        # Dataframe treatment
        returns = np.array(data['Change']).tolist()
        bond = np.array(data['US Bond']).tolist()
        stock = k * ((1+np.array(returns)).cumprod()).tolist()

        # Shift of data
        stock.insert(0,1)
        returns.insert(0,0)
        bond.insert(0,0)

        #–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––



        # Initial CPPI parameters
        cppi = k
        f = k*protected*np.exp(-rate*(len(returns)/n))
        c = cppi - f
        r = max(min(m*c,cppi),mineq*cppi)
        s = cppi - r

        # List
        portfolio = [k]
        floor = [f]
        cushion = [c]
        risky = [r]
        safe = [s]
        w_risk = [r/k]
        w_safe = [s/k]


        # Running the algorithm
        for i in range(1,len(returns)):

            # Portfolio value
            cppi = r * (1+returns[i]) + s * (1+bond[i])
            f = protected * np.exp(-rate*(len(returns)-i)/n)
            c = cppi - f
            r = max(min(m*c,cppi),mineq*cppi)
            s = cppi - r

            # Update lists
            portfolio.append(cppi)
            floor.append(f)
            cushion.append(c)
            risky.append(r)
            safe.append(s)
            w_risk.append(r/cppi)
            w_safe.append(s/cppi)

        #–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
        # Create a dataframes of all strategy's components
        total = [stock,np.round(portfolio,3),
                np.round(floor,3),np.round(cushion,3),
                np.round(risky,3),np.round(safe,3),
              np.round(w_risk,3),np.round(w_safe,3)]


        weights_cppi = pd.DataFrame(total, index= ['Stock Price', 'Portfolio','Floor',
                                  'Cushion', 'Risky Asset','Safe Asset','W_r','W_s']).transpose()
        weights_cppi = weights_cppi.iloc[:len(weights_cppi)-1,:]
        weights_cppi.index = data.index

        # Rebalancing Interval
        Test = weights_cppi[['W_r', 'W_s']]
        Test = Test.reset_index()
        Test.set_index('Date', inplace=True)
        Test = Test.resample('12M').mean()

        # Add rebalnacing interval to the dataframe
        dataset = pd.concat([data, Test], axis = 1)



        # Dataframe preparation on the trading portfolio
        dataset['W_r'].iloc[0:1] = weights_cppi['W_r'].iloc[0:1]
        dataset['W_s'].iloc[0:1] = weights_cppi['W_s'].iloc[0:1]
        dataset = dataset.fillna(method='ffill')
        dataset['Signal_Safe'] = np.where(dataset['Signal'] == 0,1,0)

        # Signal and cost list
        signal_list = np.array(dataset['Signal']).tolist()
        cost = []

        # Cost inclusion in the strategy
        for i in range(1,len(dataset)):
            if ((signal_list[i] == 0) and (signal_list[i-1] == 1)) or ((signal_list[i] == 1) and (signal_list[i-1] == 0)):
              cost.append(-0.005)
            else:
              cost.append(0)
        cost.append(0)
        dataset['Cost'] = pd.DataFrame(np.array(cost), index = dataset.index)

        # Final results
        dataset['CPPI_Risky'] = dataset['W_r'] * dataset['OBPI'] * dataset['Signal']
        dataset['CPPI_Safe'] = dataset['W_s'] * dataset['US Bond'] * dataset['Signal']
        dataset['Strategy_protection'] = dataset['CPPI_Risky'] + dataset['CPPI_Safe'] + dataset['US Bond'] * dataset['Signal_Safe'] + dataset['Cost']
        protection = pd.concat([dataset['Strategy_protection'],dataset['W_r'],dataset['W_s'],dataset['60/40'],dataset['Signal'],self.portfolio['Strategy']], axis = 1)
        self.portfolio = protection.dropna()


# In[10]:


#p = Protection(df_out,svr_strategy,rnn_strategy,'macd')
#p.OBPI(0.35)
#vis = Visualization(p.portfolio,'macd')
#vis.performance_protection()


# In[11]:


#p = Protection(df_out,svr_strategy,rnn_strategy,'macd')
#p.CPPI(1,12)
#vis = Visualization(p.portfolio,'macd')
#vis.performance_protection()


# In[ ]:





# In[12]:


class Visualization:


    def __init__(self, portfolio, strategy):
        self.portfolio = portfolio
        self.strategy = strategy


    def fig_cumulative_returns(self):
        fig = go.Figure()

        # Add first plot that represent the Cumulative return of the strategie
        fig.add_trace(go.Scatter(x=self.portfolio.index, y=((self.portfolio['Strategy'].cumsum()+1).values)*100,
                          mode='lines',
                          name="Trading Portfolio", line_color = '#4285F4'))

        # Add second plot that represent the Cumulative return of the Benchmark
        fig.add_trace(go.Scatter(x=self.portfolio.index,y = ((self.portfolio['60/40'].cumsum()+1).values)*100,
                  mode='lines',name="60/40 Portfolio", line_color ='#0F9D58'))



        # Add some layout
        fig.update_layout(title="Cumulative Returns: Trading Portfolio",
                    xaxis_title="Years",
                    yaxis_title="Cumulative Returns %", title_x=0.5,
                    paper_bgcolor="#FFFFFF",
                    plot_bgcolor="#FFFFFF",
                    height = 400,
                    legend=dict(x=0.03,
                                y=0.97,
                                traceorder='normal',
                                font=dict(size=12)),
                    template="plotly_white")

        fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='gray')
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='gray')

        return fig


    def performance_protection(self):
        fig = go.Figure()

        # Add first plot that represent the Cumulative return of the strategie
        fig.add_trace(go.Scatter(x=self.portfolio.index,
                                 y=((self.portfolio['Strategy_protection'].cumsum()+1).values)*100,
                                 mode='lines',name='Portfolio protection',
                                 line_color = '#4285F4'))

        # Add second plot that represent the Cumulative return of the Benchmark
        fig.add_trace(go.Scatter(x=self.portfolio.index,
                                 y = ((self.portfolio['60/40'].cumsum()+1).values)*100,
                                 mode='lines',
                                 name="60/40 Portfolio",
                                 line_color ='#0F9D58'))

        fig.add_trace(go.Scatter(x=self.portfolio.index,
                                 y = ((self.portfolio['Strategy'].cumsum()+1).values)*100,
                                 mode='lines',
                                 name="Original Trading Strategy",
                                 line_color ='#4B0082'))

        # Add some layout
        fig.update_layout(title="Cumulative Returns: Portfolio Protection",
                    xaxis_title="Years",
                    yaxis_title="Cumulative Returns %",
                    title_x=0.5,
                    height = 400,
                    paper_bgcolor="#FFFFFF",
                    plot_bgcolor="#FFFFFF",
                    legend=dict( x=0.03,y=0.97,
                    traceorder='normal',
                    font=dict(size=12)),
                    template="plotly_white")

        fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='gray')
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='gray')

        return fig


    def fig_technical_indicator(self):


        if self.strategy == 'macd':
            # Indicator visualization
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=self.portfolio.index, y=(self.portfolio['Indicator']).values,
                                mode='lines',name='Indicator',line_color = '#f4b042'))

            fig.add_hline(y=0,line_width=1, line_dash="dot",line_color = '#000000')

            fig.update_layout(title="MACD indicator",
                        xaxis_title="Years",
                        paper_bgcolor="#FFFFFF",
                        plot_bgcolor="#FFFFFF",
                        height = 300,
                        legend=dict(x=0.03,y=0.97,
                        traceorder='normal',
                        font=dict(size=5)),
                        template="plotly_white")

            fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='gray')
            fig.update_yaxes(showgrid=False, gridwidth=0.5, gridcolor='gray')

        elif self.strategy == 'rsi':
            # Indicator visualization
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=self.portfolio.index, y=(self.portfolio['Indicator']).values,
                                mode='lines',name='Indicator',line_color = '#f4b042'))

            fig.add_hline(y=70,line_width=1, line_dash="dot",line_color = '#000000')
            fig.add_hline(y=50, line_width=1, line_dash="dot",line_color = '#000000')
            fig.add_hline(y=30,line_width=1, line_dash="dot",line_color = '#000000')

            fig.update_layout(title="RSI indicator",
                        xaxis_title="Years",
                        paper_bgcolor="#FFFFFF",
                        plot_bgcolor="#FFFFFF",
                        height = 300,
                        legend=dict(x=0.03,y=0.97,
                        traceorder='normal',
                        font=dict(size=5)),
                        template="plotly_white")

            fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='gray')
            fig.update_yaxes(showgrid=False, gridwidth=0.5, gridcolor='gray')


        elif self.strategy == 'emv':
            # Indicator visualization
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=self.portfolio.index, y=(self.portfolio['Indicator']).values,
                                mode='lines',name='Indicator',line_color = '#f4b042'))

            fig.add_hline(y=0, line_width=1, line_dash="dot",line_color = '#000000')

            fig.update_layout(title="EMV indicator",
                        xaxis_title="Years",
                        paper_bgcolor="#FFFFFF",
                        plot_bgcolor="#FFFFFF",
                        height = 300,
                        legend=dict(x=0.03,y=0.97,
                        traceorder='normal',
                        font=dict(size=5)),
                        template="plotly_white")

            fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='gray')
            fig.update_yaxes(showgrid=False, gridwidth=0.5, gridcolor='gray')

        elif self.strategy == 'svr':
            # Indicator visualization
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=self.portfolio.index, y=(self.portfolio['Signal']).values,
                                mode='lines',name='Indicator',line_color = '#f4b042'))

            fig.add_hline(y=0.5, line_width=1, line_dash="dot",line_color = '#000000')

            fig.update_layout(title="SVR signal",
                        xaxis_title="Years",
                        paper_bgcolor="#FFFFFF",
                        plot_bgcolor="#FFFFFF",
                        height = 300,
                        legend=dict(x=0.03,y=0.97,
                        traceorder='normal',
                        font=dict(size=5)),
                        template="plotly_white")

            fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='gray')
            fig.update_yaxes(showgrid=False, gridwidth=0.5, gridcolor='gray')


        elif self.strategy == 'ann':
            # Indicator visualization
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=self.portfolio.index, y=(self.portfolio['Signal']).values,
                                mode='lines',name='Indicator',line_color = '#f4b042'))

            fig.add_hline(y=0.5, line_width=1, line_dash="dot",line_color = '#000000')

            fig.update_layout(title="ANN Signal",
                        xaxis_title="Years",
                        paper_bgcolor="#FFFFFF",
                        plot_bgcolor="#FFFFFF",
                        height = 300,
                        legend=dict(x=0.03,y=0.97,
                        traceorder='normal',
                        font=dict(size=5)),
                        template="plotly_white")

            fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='gray')
            fig.update_yaxes(showgrid=False, gridwidth=0.5, gridcolor='gray')


        elif self.strategy == 'rnn':
            # Indicator visualization
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=self.portfolio.index, y=(self.portfolio['Signal']).values,
                                mode='lines',name='Indicator',line_color = '#f4b042'))

            fig.add_hline(y=0.5, line_width=1, line_dash="dot",line_color = '#000000')

            fig.update_layout(title="RNN signal",
                        xaxis_title="Years",
                        paper_bgcolor="#FFFFFF",
                        plot_bgcolor="#FFFFFF",
                        height = 300,
                        legend=dict(x=0.03,y=0.97,
                        traceorder='normal',
                        font=dict(size=5)),
                        template="plotly_white")

            fig.update_xaxes(showgrid=False, gridwidth=0.5, gridcolor='gray')
            fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='gray')

        return fig

    def heatmap(self):

        if self.strategy == 'macd':
            fig = px.imshow(np.array(perf_macd.transpose()),
                            color_continuous_scale='Teal',
                            x = [(i+5)*2 for i in range(56)],
                            y = [(i+5)*2 for i in range(56)])

            fig.update_layout(title="In-sample performance heatmap MACD:",
                              yaxis_title="MACD Large parameter",
                              xaxis_title="MACD Small parameter",
                              paper_bgcolor="#FFFFFF",
                              plot_bgcolor="#FFFFFF",
                              height = 300,
                              legend=dict(x=0.03,y=0.97,
                              traceorder='normal',
                              font=dict(size=3)),
                              template="plotly_white")

        elif self.strategy == 'emv':
            fig = px.imshow(np.array(perf_emv.transpose()),
                            color_continuous_scale='Teal',
                            x = [i for i in range(10,121)],
                            y = [''])

            fig.update_layout(title="In-sample performance heatmap EMV:",
                              xaxis_title="EMV parameter",
                              paper_bgcolor="#FFFFFF",
                              plot_bgcolor="#FFFFFF",
                              height = 300,
                              legend=dict(x=0.03,y=0.97,
                              traceorder='normal',
                              font=dict(size=5)),
                              template="plotly_white")

        elif self.strategy == 'rsi':
            fig = px.imshow(np.array(perf_rsi.transpose()),
                            color_continuous_scale='Teal',
                            x = [i for i in range(10,121)],
                            y = [''])

            fig.update_layout(title="In-sample performance heatmap RSI:",
                              xaxis_title="RSI parameter",
                              paper_bgcolor="#FFFFFF",
                              plot_bgcolor="#FFFFFF",
                              height = 300,
                              legend=dict(x=0.03,y=0.97,
                              traceorder='normal',
                              font=dict(size=5)),
                              template="plotly_white")

        elif self.strategy == 'svr':
            fig = px.parallel_coordinates(study_svr , color = 'Rank',title="In-sample hyperparameters selection SVR",
                                          labels=study_svr.columns, color_continuous_scale = 'Teal')

            fig.update_layout(paper_bgcolor="#FFFFFF",
                      plot_bgcolor="#FFFFFF",
                      height=300,
                      template="plotly_white")

        elif self.strategy == 'ann':
            fig = px.parallel_coordinates(study_ann , color = 'Sharpe Ratio',title="In-sample hyperparameters selection ANN",
                                          labels=study_ann.columns, color_continuous_scale = 'Teal')

            fig.update_layout(paper_bgcolor="#FFFFFF",
                      plot_bgcolor="#FFFFFF",
                      height=300,
                      template="plotly_white")

        elif self.strategy == 'rnn':
            fig = px.parallel_coordinates(study_rnn , color = 'Sharpe Ratio',title="In-sample hyperparameters selection RNN",
                                          labels=study_rnn.columns, color_continuous_scale = 'Teal')

            fig.update_layout(paper_bgcolor="#FFFFFF",
                      plot_bgcolor="#FFFFFF",
                      height=300,
                      template="plotly_white")

        return fig

    def fig_alloc(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.portfolio.index, y=(self.portfolio['W_r']).values,
                            mode='lines',name='Risky Allocation',line_color = '#f44263'))

        fig.add_trace(go.Scatter(x=self.portfolio.index, y=(self.portfolio['W_s']).values,
                    mode='lines',name='Safe Allocation',line_color = '#d342f4'))


        fig.update_layout(title="Protection Allocations:",
                    xaxis_title="Years",
                    paper_bgcolor="#FFFFFF",
                    plot_bgcolor="#FFFFFF",
                    height = 300,
                    legend=dict(traceorder='normal',
                    font=dict(size=10)),
                    template="plotly_white")

        fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='gray')
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='gray')

        return fig


# In[13]:


p = Protection(df_out,svr_strategy,ann_strategy,rnn_strategy,'ann')
p.TIPP(10,12)
vis = Visualization(p.portfolio,'ann')
vis.fig_alloc()


# In[14]:


trade = Rebalancing(df_out,svr_strategy,ann_strategy,rnn_strategy)
trade.ANN()
trade.portfolio
vis = Visualization(trade.portfolio,'ann')
vis.fig_cumulative_returns()
#vis.fig_technical_indicator()
vis.heatmap()


# In[15]:


# Trading and protection dictionnaries selections
strat = [{'label': "MACD", 'value': "macd"},
         {'label': "RSI", 'value': "rsi"},
         {'label': "EMV", 'value': "emv"},
         {'label': "SVR", 'value': "svr"},
         {'label': "ANN", 'value': "ann"},
         {'label': "RNN", 'value': "rnn"}]

protec = [{'label': "OBPI", 'value': "OBPI"},
            {'label': "CPPI", 'value': "CPPI"},
            {'label': "TIPP", 'value': "TIPP"},
            {'label': "Hybrid OBPI-CPPI (HOC)", 'value': "HOC"}]


# In[16]:


# Metrics of the portfolio both trading and protection portfolios
params = {"font-size":"20px",
          "margin-left":"5px",
          "margin-top":"15px",
          "background-color":"#131313",
          "border-radius": "25px",
          "height": "75px"}

##############################################################################################################
############ Trading Portfolio metrics:
ann_returns = html.Div([html.Div([dcc.Markdown("", id="ret")],
                                 style={"font-size":params["font-size"],
                                        "margin-left":params["margin-left"],
                                        "font-weight":"bold"}),
                        html.Div([dcc.Markdown("Annual returns")], style={"margin-left":"5px"})],
                       style={"border":"2px solid",
                             "border-color":"#4285F4",
                             "font-size":"10px",
                             "background-color":params["background-color"],
                             "height": params["height"],
                             "margin-right":"5px",
                             "border-radius": params["border-radius"]})

std = html.Div([html.Div([dcc.Markdown("", id="std")],style={"font-size":params["font-size"],
                                                                       "margin-left":params["margin-left"],
                                                                      "font-weight":"bold"}),
                       html.Div([dcc.Markdown("Annual standard deviation")], style={"margin-left":"5px"})],
                      style={"border":"2px solid",
                             "border-color":"#4285F4",
                             "font-size":"10px",
                             "background-color":params["background-color"],
                             "height": params["height"],
                             "margin-left":"5px",
                             "margin-right":"5px",
                             "border-radius": params["border-radius"]})



sharpe_pf = html.Div([html.Div([dcc.Markdown("", id="sharpe_pf")],style={"font-size":params["font-size"],
                                                                       "margin-left":params["margin-left"],
                                                                      "font-weight":"bold"}),
                       html.Div([dcc.Markdown("Sharpe Ratio Trading Portfolio")], style={"margin-left":"0px"})],
                      style={"border":"2px solid",
                             "border-color":"#4285F4",
                             "font-size":"10px",
                             "background-color":params["background-color"],
                             "height": params["height"],
                             "margin-left":"5px",
                             "margin-right":"5px",
                             "border-radius": params["border-radius"]})


sharpe_b = html.Div([html.Div([dcc.Markdown("", id="sharpe_b")],style={"font-size":params["font-size"],
                                                                       "margin-left":params["margin-left"],
                                                                      "font-weight":"bold"}),
                       html.Div([dcc.Markdown("Sharpe Ratio 60/40 Portfolio")], style={"margin-left":"0px"})],
                      style={"border":"2px solid",
                             "border-color":"#4285F4",
                             "font-size":"10px",
                             "background-color":params["background-color"],
                             "height": params["height"],
                             "margin-left":"5px",
                             "margin-right":"5px",
                             "border-radius": params["border-radius"]})


mdd = html.Div([html.Div([dcc.Markdown("", id="mdd")],style={"font-size":params["font-size"],
                                                                       "margin-left":params["margin-left"],
                                                                      "font-weight":"bold"}),
                       html.Div([dcc.Markdown("Maximum Drawdown")], style={"margin-left":"5px"})],
                      style={"border":"2px solid",
                             "border-color":"#4285F4",
                             "font-size":"10px",
                             "background-color":params["background-color"],
                             "height": params["height"],
                             "margin-left":"5px",
                             "margin-right":"5px",
                             "border-radius": params["border-radius"]})

##############################################################################################################
############# Protected portfolio Metrics

ann_returns_1 = html.Div([html.Div([dcc.Markdown("", id="ret_1")],
                                 style={"font-size":params["font-size"],
                                        "margin-left":params["margin-left"],
                                        "font-weight":"bold"}),
                        html.Div([dcc.Markdown("Annual returns")], style={"margin-left":"5px"})],
                       style={"border":"2px solid",
                             "border-color":"#4285F4",
                             "font-size":"10px",
                             "background-color":params["background-color"],
                             "height": params["height"],
                             "margin-right":"5px",
                             "border-radius": params["border-radius"]})

std_1 = html.Div([html.Div([dcc.Markdown("", id="std_1")],style={"font-size":params["font-size"],
                                                                       "margin-left":params["margin-left"],
                                                                      "font-weight":"bold"}),
                       html.Div([dcc.Markdown("Annual standard deviation")], style={"margin-left":"5px"})],
                      style={"border":"2px solid",
                             "border-color":"#4285F4",
                             "font-size":"10px",
                             "background-color":params["background-color"],
                             "height": params["height"],
                             "margin-left":"5px",
                             "margin-right":"5px",
                             "border-radius": params["border-radius"]})



sharpe_pf_1 = html.Div([html.Div([dcc.Markdown("", id="sharpe_pf_1")],style={"font-size":params["font-size"],
                                                                       "margin-left":params["margin-left"],
                                                                      "font-weight":"bold"}),
                       html.Div([dcc.Markdown("Sharpe Ratio Portfolio Protection")], style={"margin-left":"0px"})],
                      style={"border":"2px solid",
                             "border-color":"#4285F4",
                             "font-size":"10px",
                             "background-color":params["background-color"],
                             "height": params["height"],
                             "margin-left":"5px",
                             "margin-right":"5px",
                             "border-radius": params["border-radius"]})


sharpe_b_1 = html.Div([html.Div([dcc.Markdown("", id="sharpe_b_1")],style={"font-size":params["font-size"],
                                                                       "margin-left":params["margin-left"],
                                                                      "font-weight":"bold"}),
                       html.Div([dcc.Markdown("Sortino Ratio Portfolio Protection")], style={"margin-left":"0px"})],
                      style={"border":"2px solid",
                             "border-color":"#4285F4",
                             "font-size":"10px",
                             "background-color":params["background-color"],
                             "height": params["height"],
                             "margin-left":"5px",
                             "margin-right":"5px",
                             "border-radius": params["border-radius"]})


mdd_1 = html.Div([html.Div([dcc.Markdown("", id="mdd_1")],style={"font-size":params["font-size"],
                                                                       "margin-left":params["margin-left"],
                                                                      "font-weight":"bold"}),
                       html.Div([dcc.Markdown("Maximum Drawdown")], style={"margin-left":"5px"})],
                      style={"border":"2px solid",
                             "border-color":"#4285F4",
                             "font-size":"10px",
                             "background-color":params["background-color"],
                             "height": params["height"],
                             "margin-left":"5px",
                             "margin-right":"5px",
                             "border-radius": params["border-radius"]})


# In[87]:


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

colors = {"background-color":"#131313",
         "color":"#ffffff"}



########## HEADER
header1 = html.Div([dcc.Markdown("**PROTECTION ON ALGORITHMIC TRADING PORTFOLIO**", style={"font-size":"35px"}),
                   dcc.Markdown("", style={"font-size":"15px"})],

                  style={"padding": "50px",
                        "background-image": "linear-gradient(#131313,#303030)",
                        #"background":"#131313",
                        "margin":"-15px 0px -0px -10px",
                        "textAlign":"center",
                        "color":"#FFFFFF"})

header2 = html.Div([dcc.Markdown("*TRADING PORTFOLIO*", style={"font-size":"20px"}),
                   dcc.Markdown("", style={"font-size":"15px"})],

                  style={"padding": "20px",
                        "background":"#4285F4",
                        "margin":"10px 0px 0px 10px",
                        "textAlign":"left",
                        "color":"#FFFFFF"})

header3 = html.Div([dcc.Markdown("*PORTFOLIO PROTECTION*", style={"font-size":"20px"}),
                   dcc.Markdown("", style={"font-size":"15px"})],

                  style={"padding": "20px",
                        "background":"#4285F4",
                        "margin":"10px 0px 0px 10px",
                        "textAlign":"left",
                        "color":"#FFFFFF"})



########## TRADING PORTFOLIO
spaces = html.Div([dcc.Markdown("", style={"height":"20px"})])

strategy = html.Div([dcc.Markdown("> **SELECTIONS FOR THE TRADING PORTFOLIO**",
                                  style={"color":colors["color"],"font-size":"20px"}),
                     dcc.Markdown("",style={"height":"20px"}),
                     dcc.Markdown('''**TRADING STRATEGY SELECTION**''',
                                  style={"color":colors["color"]}),
                     dcc.Dropdown(id="strategy",
                                  options=strat,
                                  value="macd",
                                  multi=False,
                                 style={"width":"60%"})],
                    style={"margin":"0px 0px 0px 0px",
                           "width":"100%",
                           "padding":"30px 0px 0px 0px",
                           'marginLeft' : '50px'})

input_value_1 = html.Div([dcc.Markdown("",style={"height":"20px"}),dcc.Markdown("**TRADING PARAMETER VALUES **",
                                     style={"color":colors["color"],"margin":"20px 0px 0px 50px"}),
                          dcc.Markdown("• Trading parameter 1",
                                     style={"color":colors["color"],
                                            "margin":"-10px 0px 0px 70px"}),
                        dcc.Input(id="param_1", type="text", value = 58,
                                  placeholder="Parameter 1 (ex: 12)",
                                  style={"margin":"0px 0px 0px 50px","width":"35%"})])


input_value_2 = html.Div([dcc.Markdown("• Trading parameter 2",
                                       style={"color":colors["color"],
                                              "margin":"-10px 0px 0px 70px"}),
                          dcc.Input(id="param_2", type="text", value = 76,
                                    placeholder="Parameter 2 (ex: 36)",
                                    style={"margin":"0px 0px 0px 50px","width":"35%"})])


bandeau_2_gauche = html.Div([dcc.Graph(id="heatmap")], style={"margin":"20px 0px 30px 0px"})

text_2 = dcc.Markdown('''
>
> **Trading portfolio section:**
>
In this section, you have the choice to select different trading portfolios.
For some of them, you can vary the parameters by choosing the one you want.
To adjust the selection of parameters, you can use the heatmap of the in-sample performance plot.
After the selections are made, the metrics of the trading strategy are automatically given,
as well as the performance graph and the strategy's indicator or signal value.
>
> **For more information [→ see documentation] (https://github.com/FlorentFischer/TradingPortfolioProtection/blob/0a23b032d5314b02260d54a837eafc910c1a7e21/Webapp_Dashboard/README.md)**''',
                    style={"color":colors["color"],
                           "height" : "210px",
                           "margin":"30px 0px 20px -5px",
                           "background-color":'#000000',
                           "border-radius": "10px",
                           "border":"2px solid",
                           "border-color":"#4285F4",
                           "padding" : "10px"})



bandeau_1_gauche = html.Div([strategy, input_value_1,spaces,input_value_2],
                            style={"margin":"0px 0px 0px 0px","height":"550px",
                                   "background-color":colors["background-color"],
                                   "padding":"10px 0px 0px 0px",
                                   "border-radius": "10px",
                                   "border":"2px solid",
                                   "border-color":"#4285F4"})


bandeau_1_droite = html.Div([dcc.Graph(id="cumret",
                                       style={"margin":"0px 0px 0px 0px"}),
                             dcc.Loading(id = "loading-icon-1",
                                         type="cube",
                                         fullscreen = False,
                                         style={"margin":"0px 0px 400px 0px"})])


review = html.Div([ann_returns, std, sharpe_pf, sharpe_b,mdd],
                  style={"columnCount":5, "background-color":"#303030",
                         "height":"100px","margin":"5px 0px -5px 0px",
                         "color":"white","text-align": "center"})

bandeau_2_droite = html.Div([dcc.Graph(id="indic")])

structure_1 = html.Div([bandeau_1_gauche,text_2,bandeau_2_gauche], style={"columnCount":2,"margin":"10px 0px -0px -0px"})



########## PORTFOLIO PROTECTION

############# Left Side
methods = html.Div([dcc.Markdown("> **SELECTIONS FOR THE PORTFOLIO PROTECTION**",
                                  style={"color":colors["color"],"font-size":"20px"}),
                    dcc.Markdown("**METHOD OF PROTECTION SELECTION**"),
                    dcc.RadioItems(id="Protection",options = protec,value="OBPI")],
                   style={"color":"#ffffff","margin":"10px 0px 0px 50px"})

volat = html.Div([dcc.Markdown("",style={"height":"15px"}),
                  dcc.Markdown("**PARAMETER VALUE FOR PROTECTION**",
                                style={"color":colors["color"],"margin":"10px 0px 0px 50px"}),
                  dcc.Markdown("• Target volatility",
                                     style={"color":colors["color"],
                                            "margin":"-10px 0px 0px 70px"}),
                  dcc.Input(id="volatility", type="text", value = 0.35,debounce=False,
                            placeholder="Target Volatlity (ex: 0.35)",
                            style={"margin":"0px 0px 0px 70px","width":"20%"})])

multi = html.Div([dcc.Markdown("• Multiplier",
                               style={"color":colors["color"],
                                      "margin":"-15px 0px 0px 70px"}),
                  dcc.Input(id="multiplier",type="text", value = 5,
                  placeholder="Multiplier (ex: 5)",
                  style={"margin":"0px 0px 0px 70px","width":"20%"})])

reb = html.Div([dcc.Markdown("• Rebalancing",
                               style={"color":colors["color"],
                                      "margin":"-15px 0px 0px 70px"}),
                dcc.Input(id="rebalancing",type="text", value = 3,
                  placeholder="Rebalancing (ex: 3)",
                  style={"margin":"0px 0px 0px 70px","width":"20%"})])

mini = html.Div([dcc.Markdown("• Minimum Equity Exposure",
                               style={"color":colors["color"],
                                      "margin":"-15px 0px 0px 70px"}),
                 dcc.Input(id="min_eq",type="text", value = 0.30,
                  placeholder="Minimum Equity Exposure (ex: 0.30)",
                  style={"margin":"0px 0px 0px 70px","width":"20%"})])

text_prot = dcc.Markdown('''
>
> **Portfolio protection section:**
>
In this section, you have the choice to select different portfolio protection methods.
The protection methodology chosen will be applied to the trading portfolio selected in the previous section
and with the optimal parameters found in the in-sample period for each trading strategy
(To see the optimal parameters, check the documentation). Among the protection methods that you can choose,
are available: OBPI, CPPI, TIPP, and HOC. Once the protection strategy you want to explore is selected,
input boxes of the related parameters related to the methods are activated.
You can then choose the parameters you want to test on a specific trading portfolio for a given protection method.
Once every element is specified, the web app automatically computes the portfolio protection strategy.
You can then have access to the metrics of the strategy, as well as the performance graph of the protection
and the allocation of risky and risk-free assets proposed by the protection method.
>
> **For more information [→ see documentation] (https://github.com/FlorentFischer/TradingPortfolioProtection/blob/0a23b032d5314b02260d54a837eafc910c1a7e21/Webapp_Dashboard/README.md)**''',
                    style={"color":colors["color"],
                           "height":"470px",
                           "margin":"10px 0px 20px 10px",
                           "background-color":'#000000',
                           "border-radius": "10px",
                           "border":"2px solid",
                           "border-color":"#4285F4",
                           "padding" : "50px"})

bandeau_test = html.Div([methods,volat,spaces,multi,spaces,reb,spaces,mini],
                        style={"margin":"0px 0px 0px 0px","height":"560px",
                               "background-color":colors["background-color"],
                                "border-radius": "10px",
                                "border":"2px solid",
                                "border-color":"#4285F4",
                               "padding":"10px 0px 0px 0px"})

########### Right Side

graph_protec = html.Div([dcc.Graph(id="perf_protec",style={"margin":"20px 0px 0px 0px"}),
                         dcc.Loading(id = "loading-icon-2",
                                     type="cube",
                                     fullscreen = False,
                                     style={"margin":"0px 0px 400px 0px"})])

alloc = html.Div([dcc.Graph(id="allocations",style={"margin":"0px 0px 0px 0px"})])

review_1 = html.Div([ann_returns_1, std_1, sharpe_pf_1, sharpe_b_1,mdd_1],
                    style={"columnCount":5, "background-color":"#303030",
                           "height":"100","margin":"5px 0px 5px 0px",
                           "color":"white","text-align": "center"})

structure_protec_right = html.Div([review_1,graph_protec,alloc],
                            style={"margin":"10px 0px 0px 0px"})

structure_protec = html.Div([bandeau_test,text_prot],
                            style={"columnCount":2,"margin":"10px 0px 0px 0px"})

structure_trading_right = html.Div([review,bandeau_1_droite,bandeau_2_droite],
                            style={"margin":"10px 0px 0px 0px"})

########## Dashboard
dashboard = html.Div([header1,
                      header2,
                      structure_1,
                      structure_trading_right,
                      header3,
                      structure_protec,
                      structure_protec_right],
                      style={"background":"#303030","margin":"0px -15px -0px -15px"})


app.layout = dashboard

####### CALLBACKS N°1: Trading portfolio only

@app.callback(Output("loading-icon-1", "figure"),
              Output("cumret", "figure"),
              Output("indic", "figure"),
              Output("heatmap", "figure"),
              Output("ret", "children"),
              Output("std", "children"),
              Output("sharpe_pf", "children"),
              Output("sharpe_b", "children"),
              Output("mdd", "children"),
              Output("param_2", "disabled"),
              Output("param_2", "style"),
              Output("param_1", "disabled"),
              Output("param_1", "style"),
              Input("strategy","value"),
              Input("param_1","value"),
              Input("param_2","value"))

def affichage_1(lpp_value,param1,param2):


    ################# Systematic trading strategy
    database = df_out
    trade = Rebalancing(database,svr_strategy,ann_strategy,rnn_strategy)

    if lpp_value == "macd":
        block_1 = False
        block_2 = False
        input_change_1 = {"background-color":"#FFFFFF","margin":"0px 0px 0px 50px","width":"35%"}
        input_change_2 = {"background-color":"#FFFFFF","margin":"0px 0px 0px 50px","width":"35%"}
        trade.MACD(int(param1),int(param2))


    elif lpp_value == "rsi":
        block_1 = False
        block_2 = True
        input_change_1 = {"background-color":"#FFFFFF","margin":"0px 0px 0px 50px","width":"35%"}
        input_change_2 = {"background-color":"#131313","margin":"0px 0px 0px 50px","width":"35%"}
        trade.RSI(int(param1))

    elif lpp_value == "emv":
        block_1 = False
        block_2 = True
        input_change_1 = {"background-color":"#FFFFFF","margin":"0px 0px 0px 50px","width":"35%"}
        input_change_2 = {"background-color":"#131313","margin":"0px 0px 0px 50px","width":"35%"}
        trade.EVM(int(param1))

    elif lpp_value == "svr":
        block_1 = True
        block_2 = True
        input_change_1 = {"background-color":"#131313","margin":"0px 0px 0px 50px","width":"35%"}
        input_change_2 = {"background-color":"#131313","margin":"0px 0px 0px 50px","width":"35%"}
        trade.SVR()

    elif lpp_value == "ann":
        block_1 = True
        block_2 = True
        input_change_1 = {"background-color":"#131313","margin":"0px 0px 0px 50px","width":"35%"}
        input_change_2 = {"background-color":"#131313","margin":"0px 0px 0px 50px","width":"35%"}
        trade.ANN()

    elif lpp_value == "rnn":
        block_1 = True
        block_2 = True
        input_change_1 = {"background-color":"#131313","margin":"0px 0px 0px 50px","width":"35%"}
        input_change_2 = {"background-color":"#131313","margin":"0px 0px 0px 50px","width":"35%"}
        trade.RNN()


    # Computation of metrics
    returns_ann = np.round((trade.portfolio['Strategy'].mean())*252*100,2)
    std_ann = np.round((trade.portfolio['Strategy'].std())*np.sqrt(252)*100,2)
    sharpe = np.round(returns_ann/std_ann,2)

    returns_ann_b = np.round((trade.portfolio['60/40'].mean())*252*100,2)
    std_ann_b = np.round((trade.portfolio['60/40'].std())*np.sqrt(252)*100,2)
    sharpe_b = np.round(returns_ann_b/std_ann_b,2)

    cumul = (trade.portfolio['Strategy'].dropna().cumsum()+1)
    rolling_max = np.maximum.accumulate(cumul)
    max_drawdown  = np.round(np.min(cumul/rolling_max - 1)*100,2)

    # Format
    R_ann = "{}%".format(returns_ann)
    Std_ann = "{}%".format(std_ann)
    Sharpe = "{}".format(sharpe)
    Sharpe_b = "{}".format(sharpe_b)
    MDD = "{}%".format(max_drawdown)


    # Visualization of the portfolios
    vis = Visualization(trade.portfolio,lpp_value)
    cum = vis.fig_cumulative_returns()
    indic = vis.fig_technical_indicator()
    heatmap = vis.heatmap()
    z = 3

    time.sleep(1)
    return z,cum, indic,heatmap,R_ann,Std_ann,Sharpe,Sharpe_b,MDD,block_2,input_change_2,block_1,input_change_1


####### CALLBACKS N°2: Protection of the trading portfolio

@app.callback([Output("loading-icon-2", "figure"),
              Output("perf_protec", "figure"),
              Output("allocations", "figure"),
              Output("ret_1", "children"),
              Output("std_1", "children"),
              Output("sharpe_pf_1", "children"),
              Output("sharpe_b_1", "children"),
              Output("mdd_1", "children"),
              Output("volatility", "disabled"),
              Output("volatility", "style"),
              Output("multiplier", "disabled"),
              Output("multiplier", "style"),
              Output("rebalancing", "disabled"),
              Output("rebalancing", "style"),
              Output("min_eq", "disabled"),
              Output("min_eq", "style"),],
              [Input("strategy","value"),
              Input("Protection", "value"),
              Input("volatility", "value"),
              Input("multiplier", "value"),
              Input("rebalancing", "value"),
              Input("min_eq", "value")])

def affichage_2(trading,protec,volatility,multiplier,rebalancing,min_eq):

    p = Protection(df_out,svr_strategy,ann_strategy,rnn_strategy,trading)

    if protec =='OBPI':
        p.OBPI(float(volatility))
        block_1 = False
        block_2 = True
        block_3 = True
        block_4 = True
        input_change_1={"background-color":"#FFFFFF","margin":"0px 0px 0px 50px","width":"35%"}
        input_change_2={"background-color":"#131313","margin":"0px 0px 0px 50px","width":"35%"}
        input_change_3={"background-color":"#131313","margin":"0px 0px 0px 50px","width":"35%"}
        input_change_4={"background-color":"#131313","margin":"0px 0px 0px 50px","width":"35%"}

    elif protec =='CPPI':
        p.CPPI(int(multiplier), int(rebalancing))
        block_1 = True
        block_2 = False
        block_3 = False
        block_4 = True
        input_change_1={"background-color":"#131313","margin":"0px 0px 0px 50px","width":"35%"}
        input_change_2={"background-color":"#FFFFFF","margin":"0px 0px 0px 50px","width":"35%"}
        input_change_3={"background-color":"#FFFFFF","margin":"0px 0px 0px 50px","width":"35%"}
        input_change_4={"background-color":"#131313","margin":"0px 0px 0px 50px","width":"35%"}

    elif protec =='TIPP':
        p.TIPP(int(multiplier), int(rebalancing))
        block_1 = True
        block_2 = False
        block_3 = False
        block_4 = True
        input_change_1={"background-color":"#131313","margin":"0px 0px 0px 50px","width":"35%"}
        input_change_2={"background-color":"#FFFFFF","margin":"0px 0px 0px 50px","width":"35%"}
        input_change_3={"background-color":"#FFFFFF","margin":"0px 0px 0px 50px","width":"35%"}
        input_change_4={"background-color":"#131313","margin":"0px 0px 0px 50px","width":"35%"}

    elif protec =='HOC':
        p.HOC(float(volatility),int(multiplier), float(min_eq))
        block_1 = False
        block_2 = False
        block_3 = True
        block_4 = False
        input_change_1={"background-color":"#FFFFFF","margin":"0px 0px 0px 50px","width":"35%"}
        input_change_2={"background-color":"#FFFFFF","margin":"0px 0px 0px 50px","width":"35%"}
        input_change_3={"background-color":"#131313","margin":"0px 0px 0px 50px","width":"35%"}
        input_change_4={"background-color":"#FFFFFF","margin":"0px 0px 0px 50px","width":"35%"}

    # Computation of metrics
    returns_ann = np.round((p.portfolio['Strategy_protection'].mean())*252*100,2)
    std_ann = np.round((p.portfolio['Strategy_protection'].std())*np.sqrt(252)*100,2)
    sharpe = np.round(returns_ann/std_ann,2)
    sortino = np.round((returns_ann/100)/(p.portfolio['Strategy_protection'][p.portfolio['Strategy_protection']<0].std()*np.sqrt(252)),2)

    cumul = (p.portfolio['Strategy_protection'].dropna().cumsum()+1)
    rolling_max = np.maximum.accumulate(cumul)
    max_drawdown  = np.round(np.min(cumul/rolling_max - 1)*100,2)

    # Format of the metrics
    R_ann = "{}%".format(returns_ann)
    Std_ann = "{}%".format(std_ann)
    Sharpe = "{}".format(sharpe)
    Sortino = "{}".format(sortino)
    MDD = "{}%".format(max_drawdown)

    # Visualization of the webapp graphs
    vis = Visualization(p.portfolio,trading)
    cum = vis.performance_protection()
    allocations = vis.fig_alloc()
    z = 3

    time.sleep(1)
    return z,cum,allocations,R_ann,Std_ann,Sharpe,Sortino,MDD,block_1,input_change_1,block_2,input_change_2,block_3,input_change_3,block_4,input_change_4

if __name__ == '__main__':
    app.run_server(debug=False)