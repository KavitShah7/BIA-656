#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.model_selection import TimeSeriesSplit


# In[8]:


# loading the data and having a glacne at the data
path="C:\Kavit\Stevens Institute of Technology\SEM 3\BIA 656 Advanced Data Analytics and Machine Learning\Assignment\sofr_raw_data.xls"
#df = pd.read_csv(path)
df = pd.read_excel(path)
df.info()
df.describe()


# In[9]:


#Converting to datetime type
df['DATE'] = pd.to_datetime(df['DATE'])
df['DATE']


# In[10]:


#Checking for Null Values
df.isnull().sum()


# In[14]:


#we dont need percentile values, so we are dropping those columns
df_dropped = df.drop(["1ST\n(PERCENT)","25TH\n(PERCENT)","75TH\n(PERCENT)","99TH\n(PERCENT)"], axis =1)
df_dropped.isnull().sum()

# Making a copy of the PreProcessed data 
df_dropped.to_csv('C:\Kavit\Stevens Institute of Technology\SEM 3\BIA 656 Advanced Data Analytics and Machine Learning\Assignment\sofr_.csv',index=False)


# In[15]:


def parser(s):
    return datetime.strptime(s, '%Y-%m-%d')


# In[16]:


path = 'C:\Kavit\Stevens Institute of Technology\SEM 3\BIA 656 Advanced Data Analytics and Machine Learning\Assignment\sofr_.csv'
new_df = pd.read_csv(path, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
new_df = new_df.asfreq(pd.infer_freq(new_df.index))


# In[17]:


rate = new_df['RATE\n(PERCENT)']
rate.tail(20) 


# ### Since it parse all the continuous dates but Federal Bank remains close on public holidays and sat,sun it shows NaN values for those days. For better visualization of data we are going to fill the Nan values by it's previous day's rate. 
# 

# In[18]:


# imputed the null values with Forward Fill as is select the value from the previous cell which is filled
rate = rate.fillna(method='ffill')
# Log Return of the SOFR and dropping Null values 
rate = rate.pct_change().dropna()


# In[19]:


#plt.figure(figsize=(10,4))
#plt.plot(series)
#plt.title('Simulated GARCH(2,2) Data', fontsize=20)


# In[20]:


def plot_series(series):
    plt.figure(figsize=(12,6))
    plt.plot(rate, color='red')
    plt.ylabel('Search Frequency for Rate', fontsize=16)

    for year in range(2018, 2021):
        for month in range(1,13):
            plt.axvline(datetime(year,month,1), linestyle='--', color='k', alpha=0.5)


# In[21]:


plot_series(rate)
#by the plot we can see some high volatility in behaviour of graph, there is some arch effect 


# ### Normalizing 

# In[22]:


avg, dev = rate.mean(), rate.std()
rate = (rate - avg) / dev


# In[23]:


plot_series(rate)
plt.axhline(0, linestyle='--', color='k', alpha=0.3)


# #### Taking First difference to remove the trend

# In[24]:


rate = rate.diff().dropna()


# In[25]:


plot_series(rate)
plt.axhline(0, linestyle='--', color='k', alpha=0.3)


# #### Removing Seasonality

# In[26]:


month_avgs = rate.groupby(rate.index.month).mean()


# In[27]:


month_avgs


# In[28]:


rate_month_avg = rate.index.map(lambda d: month_avgs.loc[d.month])


# In[29]:


rate_month_avg


# In[30]:


rate = rate - rate_month_avg


# In[31]:


plot_series(rate)
plt.axhline(0, linestyle='--', color='k', alpha=0.3)


# In[32]:


plot_pacf(np.array(rate)**2)
plt.show()

#reason to garch(2,2) model 


# In[33]:


plot_acf(np.array(rate)**2)
plt.show()
##there is seroial corelationat lag 1 


# In[ ]:




