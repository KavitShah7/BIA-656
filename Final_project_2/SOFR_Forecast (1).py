#!/usr/bin/env python
# coding: utf-8

# In[139]:


import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.model_selection import TimeSeriesSplit
import statsmodels.api as sm


# In[140]:


def parser(s):
    return datetime.strptime(s, '%Y-%m-%d')


# In[141]:


path = 'C:/Hariom Mehta/Academics/Masters MIS docs/SEM 3/BIA 656/Final Project/sofr_.csv'
new_df = pd.read_csv(path, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
new_df = new_df.asfreq(pd.infer_freq(new_df.index))


# In[212]:


rate = new_df['RATE\n(PERCENT)']
rate.tail(20) 
rate = rate.fillna(method='ffill')
rate = rate.pct_change().dropna()


# In[143]:



plt.figure(figsize=(10,4))
plt.plot(rate)
plt.ylabel('Pct Return', fontsize=16)
plt.title('SOFR Returns', fontsize=20)


# In[144]:


#ljung box test
res = sm.tsa.ARMA(rate, (1,1)).fit(disp=-1)
#sm.stats.acorr_ljungbox(res.resid, lags=[10])
sm.stats.acorr_ljungbox(res.resid, lags=[5])
# pvalue is high, reject null hypo, there is no serial corelation 


# In[145]:


sm.stats.diagnostic.het_arch(res.resid)
#Lagrange multiplier test statistic, p-value for Lagrange multiplier test

#fstatistic for F test, alternative version of the same test based on F test for the parameter restriction

#p value is very low, reject nul hypo, there is arch effects


# In[146]:


plot_pacf(rate**2)
plt.show()


# In[147]:


model = arch_model(rate, p=1, q=1, rescale=False)
model_fit = model.fit()
model_fit.summary()

AIC:	-2204.30,OMEGA, alpha[1],beta[1] is significant


# In[148]:


model = arch_model(rate, p=2, q=1, rescale=False)
model_fit = model.fit()
model_fit.summary()

#AIC:	1113.88 , alpha[1] is significant, but alpha[2], beta[1] is not significant


# In[150]:


model = arch_model(rate, p=2, q=2)
model_fit = model.fit()
model_fit.summary()


# In[151]:


model = arch_model(rate, p=2, q=3)
model_fit = model.fit()
model_fit.summary()


# In[153]:


model = arch_model(rate, p=1, q=1)
model_fit = model.fit()
model_fit.summary()


# In[154]:


predictions = model_fit.forecast(horizon=10)


# In[159]:


rolling_predictions = []
test_size = 365

for i in range(test_size):
    train = returns[:-(test_size-i)]
    model = arch_model(train, p=1, q=1,rescale=False)
    model_fit = model.fit(disp='off')
    pred = model_fit.forecast(horizon=1)
    rolling_predictions.append(np.sqrt(pred.variance.values[-1,:][0]))


# In[160]:



rolling_predictions = pd.Series(rolling_predictions, index=returns.index[-365:])


# In[162]:



plt.figure(figsize=(10,4))
true, = plt.plot(returns[-365:])
preds, = plt.plot(rolling_predictions)
plt.title('SOFR Volatility Prediction - Rolling Forecast', fontsize=20)
plt.legend(['True Returns', 'Predicted Volatility'], fontsize=16)


# In[208]:


from scipy.stats import kurtosis, skew


# In[209]:


print( 'excess kurtosis of normal distribution (should be 0): {}'.format( kurtosis(rate) ))
print( 'skewness of normal distribution (should be 0): {}'.format( skew(rate) ))


# In[220]:


skewt_gm = arch_model(rate, p = 1, q = 1, mean = 'constant', vol = 'GARCH', dist = 'skewt')
skewt_gm = skewt_gm.fit()


# In[221]:


skewt_gm.summary()


# In[218]:



pred =skewt_gm.forecast(horizon=10)


# In[219]:


rolling_predictions = []
test_size = 365

for i in range(test_size):
    train = returns[:-(test_size-i)]
    model = arch_model(train, p = 1, q = 1, vol = 'GARCH', dist = 'skewt',rescale=False)
    model_fit = model.fit()
    pred = model_fit.forecast(horizon=1)
    rolling_predictions.append(np.sqrt(pred.variance.values[-1,:][0]))
    



# In[223]:


rolling_predictions = pd.Series(rolling_predictions, index=returns.index[-365:])


# In[224]:


plt.figure(figsize=(20,8))
true, = plt.plot(returns[-365:])
preds, = plt.plot(rolling_predictions)
plt.title('Volatility Prediction - Rolling Forecast', fontsize=20)
plt.legend(['True Returns', 'Predicted Volatility'], fontsize=16)


# In[ ]:




