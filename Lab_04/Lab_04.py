#!/usr/bin/env python
# coding: utf-8

# # Lab 04

# In[1]:


import os
import numpy as np
import pandas as pd
import math
import matplotlib.pylab as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style='ticks', palette='Set2')


# ## 1. Load Data and show high-level statistical info about the data frame's columns

# In[2]:


path = "C:\Kavit\Stevens Institute of Technology\SEM 3\BIA 656 Advanced Data Analytics and Machine Learning\Assignment\Lab04\Concrete_Data.csv"
df = pd.read_csv(path)[["Cement", "Blast_Furnace_Slag", "Fly_Ash", "Water", "Superplasticizer","Coarse_Aggregate","Fine_Aggregate","Age","Concrete_Compressive_Strength"]]

#Taking a look at the data
df.head(5)


# In[3]:


df.describe()


# In[4]:


df.info()


# ## 2. How many rows have a compressive strength > 40 MPa? 

# In[5]:


df1 = df[df.Concrete_Compressive_Strength > 40.00]
df1
len(df1)
print("Ans: " + str(len(df1)) + " rows have compressive strength > 40 MPa")


# ##  3. Plot histograms of Plot_Aggregate and Fine_Aggregate

# In[6]:


df.hist("Coarse_Aggregate")
df.hist("Fine_Aggregate")


# ##  4.  Make a plot comparing compressive strength to age

# In[7]:


df.plot(kind="scatter", x="Concrete_Compressive_Strength",y="Age")


# In[8]:


df.plot(kind="density", x="Concrete_Compressive_Strength",y="Age")


# ## 5. Make a plot comparing compressive strength to age for only those rows with < 750 fine aggregate.

# In[9]:


df2=df[df.Fine_Aggregate < 750.0]
df2


# ## 6.  Try to build a linear model that predicts compressive strength given the other available fields.

# In[10]:


from sklearn import linear_model 

# Choosing the linear model
linear_model=linear_model.Lasso(alpha=0.01)

#Preparing the model by setting up target and features
features=["Cement", "Blast_Furnace_Slag", "Fly_Ash", "Water", "Superplasticizer","Coarse_Aggregate","Fine_Aggregate","Age"]
target="Concrete_Compressive_Strength"

# Fitting the linear model
linear_model.fit(df[features],df[target])


#Coefficients of the linear model
pd.DataFrame([dict(zip(features,linear_model.coef_))])


# ## 7.Generate predictions for all the observations and a scatterplot comparing the predicted compressive strengths to the actual values.
#  

# In[11]:


preds =linear_model.predict(df[features])
predictions_df = df.assign(predictions=preds)
predictions_df[["Concrete_Compressive_Strength", "predictions"]]


# In[12]:


#A scatterplot comparing the predicted compressive strengths to the actual values.
predictions_df.plot(kind="scatter",x="predictions",y="Concrete_Compressive_Strength")


# #### The model gives us many outliers and is not accurate enough as we can see in the scatter plot.
