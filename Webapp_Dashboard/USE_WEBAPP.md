# Web App Dashboard Documentation


1. Trading Portfolio Elements
2.  Portfolio Protection Elements
3.   Technical info 




# 1. Trading Portfolio Elements:


### A. Input box:

In this first part of the webapp, you have the possibility to explore the 6 strategies from the MASTER Thesis.
First you can select wich strategy you want to study
Then depending on the selected trading strategy, you test a different selection of parameter from the optimal one found in the Master Thesis, to see the effectiveness of the parameter selection. 
Input boxes for parameter selections are activated or deactivated depending on the strategy selected. 

### B. Help Box:

This box provides some advice to use the portfolio protection section of the dashboard

### C. In-sample performance analysis:

### D. Metrics of the strategy:

### E. Performance of the trading strategy:

### F. Indicator value:




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

This box provides some advice to use the portfolio protection section of the dashboard

### C. Metrics of the strategy:

### D. Performance of the protection strategy:

### E. Indicator value:



