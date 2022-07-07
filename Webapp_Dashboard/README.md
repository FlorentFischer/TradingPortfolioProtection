# Web App Dashboard Documentation


1. Trading Portfolio Elements
   - Selection Box 
   - Help Box
   - In-sample performance analysis
   - Metrics of the strategy
   - Performance of the trading strategy
   - Indicator or Signal value
2. Portfolio Protection Elements
   - Selection Box 
   - Help Box
   - Metrics of the protection
   - Performance of the protection strategy
   - Protection method allocations
3. Technical info 


<br />
<br />

# 1. Trading Portfolio Elements:


### A. Selection box:

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

For the parameter selection of the Trading portfolio section, you can change the parameters for some trading strategies (not for machine and deep learning). It allows the user to explore the results and see if the optimal parameter found in the in-sample frame is consistent in the out-sample frame. After typing the parameter desired, the associated strategy is automatically computed by the web app. 

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

This box aims to provide some advice to the user of the dashboard. Indeed it helps the user to understand the different elements present in the first section (Trading Portfolio). 

### C. In-sample performance analysis:

This first graphical element is a guide for the user to choose and explore the different possible parameters for the trading strategies. It just has an informational purpose and gives only an indication of results obtained in the in-sample framework. It could be useful to choose the optimal parameters of the technical indicators trading strategies, but for the machine and deep learning strategies, it has only utility to inform the user if he wants to reproduce the strategies. For every plot, the objective functions of the heatmaps are the Sharpe ratio function, depending on the strategy parameters. 

|Strategy|Graph|
|---|---|
|MACD|Heatmap|
|RSI|Heatmap|
|EMV|Heatmap|
|SVR|Parallel coordinate plot|
|ANN|Parallel coordinate plot|
|RNN|Parallel coordinate plot|

### D. Metrics of the strategy:

Above the performance graph of the trading strategy, we can see several metrics of the trading strategy selected previously. They are essential for the reading of the trading strategy in addition to the performance visualization. 

Among the metrics available you have: 

|Metrics|
|---|
|Annual returns|
|Annual standard deviation|
|Annual Sharpe ratio Trading Portfolio|
|Annual Sharpe ratio 60/40 Portfolio|
|Maximum Drawdown|


### E. Performance of the trading strategy:

This graph aims to plot the performance of the Trading strategy and the 60/40 Portfolio by showing the cumulative returns of the strategy and the related benchmark. Here, you will have the opportunity to see the efficiency of the trading strategy against the 60/40 Portfolio depending on the parameters selected previously when possible. 

### F. Indicator or Signal value:

This graph aims to plot the Technical Indicator or Signal value of the Trading strategy depending on the trading strategy chosen previously. It will help you to understand how and when the portfolio is traded, and see how it is reflected in the performance of the strategy on the performance graph above. 

<br />
<br />

# 2. Portfolio Protection Elements:


### A. Protection strategy selection

> #### Protection method selection
Here you can select one of the four methods of portfolio protection implemented in the Master Thesis. When selected, the protection method will be applied to the optimal strategy selected in the trading portfolio part. Noticed that the protection does not depend on the selected parameter from the trading portfolio part. 

Among the methods available, you can select:
- OBPI: Option-based portfolio protection
- CPPI: Constant portfolio protection insurance
- TIPP: Time invariant portfolio protection
- HOC: Hybrid OBPI and CPPI

 > #### List of the available portfolio protection methods:
| Accronym | Portfolio Protection Method | 
|---|---|
|OBPI|Option-based portfolio protection|
|CPPI|Constant portfolio protection insurance|
|TIPP|Time invariant portfolio protection|
|HOC|Hybrid OBPI and CPPI|


> #### Parameters selection

Here you can change the parameters of the protection methods. Some parameters are already set by defaults but didn't correspond to a particular optimal method. The possibility of a parameter depends on the method selected above. Non-essential input boxes are deactivated, remaining activated only necessary input boxes. 

Among the parameters to be modified you have:
- Target volatility: This parameter is useful for OBPI and HOC methods.
- Multiplier: This parameter is useful for CPPI, TIPP, and HOC methods. 
- Rebalancing: This parameter is useful for CPPI and TIPP methods.
- Minimum equity exposure: This parameter is useful for the HOC method.

| Parameters | Methods implied | Range of variation | Recommended parameter |
|---|---|---|---|
| Target volatility | OBPI / HOC | 0 to 1 | ~0.3 - 0.35 |
| Multiplier | CPPI / TIPP / HOC | 1 to 10+ | 1 |
| Multiplier | CPPI / TIPP | 1 to 36+ | 12 |
| Minimum equity exposure | HOC | 0 to 1 | ~0.3 - 0.5 |

### B. Help Box:

This box aims to provide some advice to the user of the dashboard. Indeed it helps the user to understand the different elements present in the second section (Portfolio Protection). 

### C. Metrics of the strategy:

Above the performance graph of the portfolio protection, we can see several metrics of the protection method selected previously. They are essential for the well reading of the protection method implemented in addition to the performance visualization of the protection method. 

Among the metrics available you have: 

|Metrics|
|---|
|Annual returns|
|Annual standard deviation|
|Annual Sharpe ratio Portfolio Protection|
|Annual Sortino ratio Portfolio Protection|
|Maximum Drawdown|

### D. Performance of the protection strategy:

This graph aims to plot the performance of the portfolio protection strategy of a given trading strategy, the trading strategy, and the 60/40 portfolio by showing their cumulative returns. Here, you will have the opportunity to see the efficiency of the portfolio protection strategy against the trading strategy, and against the 60/40 portfolio depending on the parameters selected previously. 

### E. Protection method allocations:

This graph aims to plot the weight allocations of the protection method selected previously. It will help you to understand how the allocations between the risky and the risk-free assets are distributed. This graph doesn't plot the allocation of the trading strategy adaption to the protection method but the allocation recommended by the protection method only. In-sample hyperparameters selection RNN


