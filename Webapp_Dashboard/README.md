# Web App Dashboard Documentation


1. Trading Portfolio Elements
2.  Portfolio Protection Elements
3.   Technical info 


<br />
<br />

# 1. Trading Portfolio Elements:


### A. Input box:

> #### Trading strategy selection

In this first part of the webapp, you have the possibility to explore the 6 strategies from the Master Thesis. Here you just have to select one of the 6 strategies proposed, and then the webapp automatically integred it in the different graph and metrics of the webapp. After specifying a trading strategy, all of the webapp will consider this strategy for the different analysis graphs and metrics, and also consider it for the portfolio protection part. 

 > #### List of the available trading strategies:
| Accronym | Trading Strategy | 
|---|---|
|MACD|Moving Average Convergence Divergence|
|RSI|Relative Strength Index|
|EMV|Ease of Movment Value|
|SVR|Support Vector Regression|
|ANN|Artificial Neural Network|
|RNN|Recurrent Neural Network|


> #### Parameter selection

For the parameter selection of the Trading portfolio section, you have the possibility to change the parameters for some trading strategies (not for machine and deep learning). It allows the user to explore the results and see if the optimal parameter found in the in-sample frame is consistent in the out-sample frame. After typing the parameter desired, the associated strategy is automatically computed by the webapp. 

Among the parameters to be modified you have:
  - Parameter 1: For MACD, RSI, and EMV trading strategies 
  - Parameter 2: For MACD trading strategy
  
 > #### Guide of parameters variations:
 
 |Parameter | Trading Strategy involved | Optimal Parameter |
 |---|---|---|
 |Parameter 1 |MACD | 56 |
 |Parameter 1 |RSI | 67 |
 |Parameter 1 |EMV | 31 |
 |Parameter 2 |MACD | 74 |
 

### B. Help Box:

This box aims to provide some advice to the user of the dashboard. Indeed it help the user to understand the diffrent element present in the first section (Trading Portfolio). 

### C. In-sample performance analysis:

This first graphical element is guide for the user to choose and explore the different possible parameters for the trading strategies. It just have an informal purpose and gives only an indication of results obtained in the in-sample framework. It could be useful for to choose the optimal parameters of the technical indicators trading strategies, but for the machine and deep learning strategies it have only an utility to inform the user if he want to reproduces the strategies. For every plot, the objective functions of the heatmaps are the Sharpe ratio function, dpending on the strategy parameters. 

|Strategy|Graph|
|---|---|
|MACD|Heatmap|
|RSI|Heatmap|
|EMV|Heatmap|
|SVR|Parallel coordinate plot|
|ANN|Parallel coordinate plot|
|RNN|Parallel coordinate plot|

### D. Metrics of the strategy:

Above the perfomance graph of the trading strategy, we can see several metrics of the strategy selected previously. It is essential for the reading of the trading strategy in addition with the performance visualization. 

Among the metrics available you have: 

|Metric|
|---|
|Annual returns|
|Annual volatility|
|Annual Sharpe ratio|
|Annual Sortino ratio|
|Maximum Drawdown|


### E. Performance of the trading strategy:

This graph aims to plot the performance of the Trading strategy and the S&P500 by diplaying the cumulative returns of the strategy and the related benchmark. Here, you will have the opprtunity to see the efficency of the trading strategy against the S&P500 depending on the parameters selected previously when possible. 

### F. Indicator or Signal value:

This graph aims to plot the Indicator or Signal value of the Trading strategy depending on the trading strategy choosen previously. It will help you to understand how and when the portfolio is traded, and see how it is reflectd to the performance of the strategy on the performance graph above. 

<br />
<br />

# 2. Portfolio Protection Elements:


### A. Protection strategy selection

> #### Protection method selection
Here you have the possibility to select one of the four methods of portfolio protection implemented in the Master Thesis. When selected, the protection method will be applied to the optimal strategy selected in the trading portfolio part. /!\ Noticed that the protection do not depend on potential parameter selection from trading portfolio part /!\. 

Among the methods available, you can select:
- OBPI: Option based portfolio protection
- CPPI: Constant portfolio protection inssurance
- TIPP: Time invariant portfolio protection
- HOC: Hybrid OBPI and CPPI

> #### Parameters selection

Here you have the possibility to change the parameters of the protection methods. Some parameters are already set by defaults but didn't correspond to particular optimal method. The possibility of parameter depends on the method selected above. Non-essential input box are deactivated, remaining activated only necessary input box. 

Among the parameters to be modified you have:
- Target volatility: This parameter is useful for OBPI and HOC methods.
- Multiplier: This parameter is useful for CPPI, TIPP, and HOC methods. 
- Rebalancing: This parameter is useful for CPPI and TIPP methods.
- Minimum equity exposure: This parameter is useful for HOC method.

### B. Help Box:

This box aims to provide some advice to the user of the dashboard. Indeed it help the user to understand the diffrent element present in the second section (Portfolio Protection). 

### C. Metrics of the strategy:

Above the perfomance graph of the portfolio protection, we can see several metrics of the strategy selected previously. It is essential for the reading of the protection method implemennted in addition with the performance visualization of the protection method. 

Among the metrics available you have: 

|Metric|
|---|
|Annual returns|
|Annual volatility|
|Annual Sharpe ratio|
|Annual Sortino ratio|
|Maximum Drawdown|

### D. Performance of the protection strategy:

### E. Indicator value:



