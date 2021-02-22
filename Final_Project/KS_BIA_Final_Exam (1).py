#!/usr/bin/env python
# coding: utf-8

# # Final Exam - BIA 656 

# ### Name : Kavit Shah
# ### CWID: 10452991

# ###  1. Include a brief description of the stock under study and possible factors that may affect its performance. 

# DUK stands for Duke Energy Corporation which operates an energy company in United States. It operates in 3 segments Electric Utilities and Infrastructure and sells electricity in the Carolinas, Florida, and the Midwest; and uses coal, hydroelectric, natural gas, oil, renewable sources, and nuclear fuel to generate electricity.
# 
# The stock summary is as follows :
# 
# #### Day's Range:	90.14 - 92.67
# #### 52 Week Range:	62.13 - 103.79
# #### Volume:	3,172,227
# #### Avg. Volume:	3,395,380
# #### Market Cap:	67.083B
# #### Beta (5Y Monthly):	0.21
# #### PE Ratio (TTM):	33.40
# #### EPS (TTM):	2.73
# #### Forward Dividend & Yield:	3.86 (4.18%)
# #### 1y Target Est:	98.1

# ### 2. What precisely is the forecasting problem? How do you plan to calculate it? What is the target variable in your machine learning algorithm? 

# The forecasting problem here is to predict stock prices and the day's trend(which can go upwards or downwards) for 2019 as well as training the data from 2014-2018.Calculation will be done using the machine learning algorithms and predicting the right target variable for the model. The target variable in the machine learning algorithm will be Trend Column which I will be calculating based on the RET column and storing it in the ret_rate_diff column
# 
# Ultimately it is the return trend that we want, so for returnong the trend purpose I have introduced the Trend Column

# ### 3. Which three forecasting machine learning algorithms do you think are appropriate for this problem domain and why? How do you plan to select the best algorithm? Justify your answer.
#  

# The three forecasting machine learning algorithms appropriate for this domain are:
# LogisticRegression,
# Decision Tree,
# AdaBoost
# 
# Logistic Regression is good for the data because it not only provides the the measure of how correct the co-efficient is but also the direction of the assocaiton of the variable, ehich I need for my model. It is quick and makes no assumption of the distribution of the classes.
# 
# Decison Tree does not need scaling and normalizing of data as other models do and is easy to implement.
# 
# AdaBoost provides with many parameters to tune according to the model and can be used to spot the weak learner which help us tune the model manually for better performance  

# ### 4. What features would you use? You do not have to use all the variables included in the dataset, and you could also use additional variables. Would you modify or correct your data in any way?
# ### If you plan to introduce changes to any variable, you must be very specific and tell what variable(s) will change and how. 

# The features I'll use are alpha, b_mkt, b_hml, b_smb, tvol since they affect the model the most. I personally think there is no feature I would like to add. The given data is sufficient to determine the model.
# Unused data column will be dropped for the model calibration.

# ### 5. How will you calibrate your model and evaluate whether your model has captured anygeneralizable knowledge? Explain your method, and justify the metric(s) that you propose to employ

# The parameters of machine learning algorithm which we select for the final modelwill be tuned by GridSearchCV
# So, the calibration of the model will be done by GridSearchCV. The algorithm will capture the generalizable knowledge and will tune the parameters accordingly which will result in high accuracy for that model than before.

# ### 6.  Compare two graphs that you could use to evaluate the different algorithms' performance and explain how they can help evaluate your models and select the best method. Select one graph and justify your answer.

# We are predicting the trend and not only the values so the curves that help us in this are ROC and Cumultaive curve. It helps us predict the trend. Less area under the plot means that the model preformnace is not that great. ROC plots are easier to understand. 

# ### 7.  How can you rank the features' importance?

# The Logistic Regression classifier has feature importance feature  which helps us to rank the features importance using the model co-efficients  

# ###  8. Once you put your system in production, would you leave it to work alone, or what else can you do?

# The system even though is complete and in production, there are always techniques to improve the model further, continuous monirtoing the model. The stocks of the company can be affected in many ways such as a natural disaster, pandemics, market crash , political events like election and many more such events around the world. We can make the model predict more accurate results and based on that results we can make our buying and selling of stocks more relaible.

# # Part-2

# In[29]:


#Importing Libraries
import os
import numpy as np
import pandas as pd
import math
import matplotlib.pylab as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as mpl
from sklearn.preprocessing import scale
from matplotlib import style
import pylab as pl
import matplotlib.dates as mdates
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel 
from pandas.plotting import lag_plot
from matplotlib import pyplot
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = 15, 12



# In[30]:


# Loading the Dataset
df= pd.read_excel(r'C:\Kavit\Stevens Institute of Technology\SEM 3\BIA 656 Advanced Data Analytics and Machine Learning\Assignment\DUK.xlsx')
 
df.head(10)


# In[31]:


# Having a look at the datatype and the null values of the columns
df.info()


# In[32]:


#dropping rows that are not needed such as n, PERMNO, ivol, TICKER as they are of no use to us
df1 = df.drop(['TICKER', 'PERMNO', 'n','exret'], axis = 1)
df1.head() 


# In[33]:


#Forming a new column ret_rate_diff from RET for calculating the trend and mapping on the graph. 
i=1
ret_rate_diff=[0]
trend = [1]
while i<len(df1):
    ret_rate_diff.append(df1.iloc[i]['RET']-df1.iloc[i-1]['RET'])
    if ret_rate_diff[i] > ret_rate_diff[i-1]:
      trend.append(1)
    else:
      trend.append(-1)  
    i+=1
    
df1['Trend'] = trend

df1.head(5)


# In[34]:


#sorting the date in ascending order
df1 = df1.sort_values('DATE')
df1


# In[35]:


#Converting the date dataype to datetime for splitting the dataset into training(2014-2018) and testing(2019) 
df1['DATE']=pd.to_datetime(df1['DATE'],format='%Y%m%d')
df1.info()


# In[36]:


#splitting the data into training(2014-2018) and testing(2019)
split_date = pd.Timestamp("2018-12-31")
split_date.date()

training = df1.loc[df1['DATE'] <= split_date]
test = df1.loc[df1['DATE'] > split_date]

# print(training)
# print(test)


# In[37]:


# Having a galnce at the data realtions (raw information)
import statsmodels.formula.api as smf 
df1_model = smf.ols('Trend~   alpha + b_smb + b_hml + b_mkt + ivol + R2 + tvol  ', data = df1).fit()
print(df1_model.summary())


# In[38]:


#Splitting into Train and Test of features and target variable 
X_train = training.iloc[:,np.r_[2,3,4,5,7]]
X_train = X_train.values
y_train = training.iloc[:,9]
y_train = y_train.values

X_test = test.iloc[:, np.r_[2,3,4,5,7]]
X_test = X_test.values
y_test = test.iloc[: ,9]
y_test= y_test.values
df1.info()


# In[39]:


#LogisticRegression of the model
#Even though the accuracy is more the hyperparameter tuning is limited
model=LogisticRegression()
clf=model.fit(X_train, y_train)
pred_clf=clf.predict(X_test)

#print(pred_clf)
#y_pred_lr=model.predict(X_test)

from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X_train, y_train, scoring="accuracy", cv=10)

print ("Cross Validated Accuracy: %0.5f +/- %0.5f" % (scores.mean(), scores.std()))


# In[40]:


#ROC curve
#Get the probability of Y_test records being = 1
Y_test_probability_1 = model.predict_proba(X_test)[:, 1]

#Use the metrics.roc_curve function to get the true positive rate (tpr) and false positive rate (fpr)
fpr, tpr, thresholds = metrics.roc_curve(y_test, Y_test_probability_1)
    
#Getting the area under the curve (AUC)
auc = scores.mean()

#Plotting the ROC curve
plt.plot(fpr, tpr, label="AUC " + str(round(auc, 2)))
plt.xlabel("False positive rate (fpr)")
plt.ylabel("True positive rate (tpr)")
plt.plot([0,1], [0,1], 'k--', label="Random")
plt.legend(loc='best')


# In[41]:


#Cumulative Curve
order = np.argsort(Y_test_probability_1)[::-1]
Y_test_probability_1_sorted = Y_test_probability_1[order]
Y_test_sorted = np.array(y_test)[order]

#Building the cumulative response curve
x_cumulative = np.arange(len(Y_test_probability_1_sorted)) + 1
y_cumulative = np.cumsum(Y_test_sorted)

scale =100

#Rescaling
x_cumulative = np.array(x_cumulative)/float(x_cumulative.max()) * scale
y_cumulative = np.array(y_cumulative)/float(y_cumulative.max()) * scale
plt.plot(x_cumulative, y_cumulative,label=model)


# # Decision Tree

# In[42]:


#Implementing the decision tree algorithm
from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier()

#To provide the data and to fit the model
d_t=decision_tree.fit(X_train,y_train)

# to check the cross_val_score
scores = cross_val_score(d_t, X_train, y_train, scoring="accuracy", cv=10)

print ("Cross Validated Accuracy: %0.5f +/- %0.5f" % (scores.mean(), scores.std()))


# In[43]:


#ROC curve
Y_test_probability_1 = d_t.predict_proba(X_test)[:, 1]

#Use the metrics.roc_curve function to get the true positive rate (tpr) and false positive rate (fpr)
fpr, tpr, thresholds = metrics.roc_curve(y_test, Y_test_probability_1)
    
#Getting the area under the curve (AUC)
auc = scores.mean()

#Plotting the ROC curve
plt.plot(fpr, tpr, label="AUC " + str(round(auc, 2)))
plt.xlabel("False positive rate (fpr)")
plt.ylabel("True positive rate (tpr)")
plt.plot([0,1], [0,1], 'k--', label="Random")
plt.legend(loc='best')


# In[44]:


#Cumulative curve
order = np.argsort(Y_test_probability_1)[::-1]
Y_test_probability_1_sorted = Y_test_probability_1[order]
Y_test_sorted = np.array(y_test)[order]

#Building the cumulative response curve
x_cumulative = np.arange(len(Y_test_probability_1_sorted)) + 1
y_cumulative = np.cumsum(Y_test_sorted)

scale =100

# Rescaling
x_cumulative = np.array(x_cumulative)/float(x_cumulative.max()) * scale
y_cumulative = np.array(y_cumulative)/float(y_cumulative.max()) * scale
plt.plot(x_cumulative, y_cumulative,label=d_t)


# # AdaBoost

# In[45]:


#implementing the AdaBoost Algorithm
from sklearn.ensemble import AdaBoostClassifier

adaboost= AdaBoostClassifier()
                                                                   
#To provide the data and to fit the model
adaboost = adaboost.fit(X_train,y_train)

# to check the cross_val_score
scores = cross_val_score(adaboost, X_train, y_train, scoring="accuracy", cv=10)

print ("Cross Validated Accuracy: %0.5f +/- %0.5f" % (scores.mean(), scores.std()))


# In[46]:


#plotting the graph for understanding
window= 12
y_pred = adaboost.predict(X_test)
test["y_pred"] = y_pred


rolling_mean = test["y_pred"].rolling(window).mean()
test["Rolling Average"] = rolling_mean

plt.figure(figsize=(15,5))
plt.title("Moving average\n window size = {}".format(window))

plt.plot(test["DATE"], test["y_pred"] , label="Predicted values")
plt.plot(test["DATE"], test["Rolling Average"], "g", label="Rolling mean trend")

plt.plot(test["DATE"], y_test , label="Actual values")
#plt.xticks(test["y_pred"],test["DATE"])
plt.legend(loc="upper left")
plt.grid(True)
plt.show()


# In[47]:


#ROC Curve
Y_test_probability_1 = adaboost.predict_proba(X_test)[:, 1]

#Use the metrics.roc_curve function to get the true positive rate (tpr) and false positive rate (fpr)
fpr, tpr, thresholds = metrics.roc_curve(y_test, Y_test_probability_1)
    
#Getting the area under the curve (AUC)
auc = scores.mean()

#Plotting the ROC curve
plt.plot(fpr, tpr, label="AUC " + str(round(auc, 3)))
plt.xlabel("False positive rate (fpr)")
plt.ylabel("True positive rate (tpr)")
plt.plot([0,1], [0,1], 'k--', label="Random")
plt.legend(loc='best')


# In[48]:


#Cumulative Curve
order = np.argsort(Y_test_probability_1)[::-1]
Y_test_probability_1_sorted = Y_test_probability_1[order]
Y_test_sorted = np.array(y_test)[order]

# Building the cumulative response curve
x_cumulative = np.arange(len(Y_test_probability_1_sorted)) + 1
y_cumulative = np.cumsum(Y_test_sorted)

scale =100

# Rescaling
x_cumulative = np.array(x_cumulative)/float(x_cumulative.max()) * scale
y_cumulative = np.array(y_cumulative)/float(y_cumulative.max()) * scale
plt.plot(x_cumulative, y_cumulative,label=adaboost)


# In[49]:


#Implementing the GridSearchCV
from sklearn.model_selection import GridSearchCV


p ={'base_estimator__max_depth':[1,100],
    'base_estimator':[DecisionTreeClassifier(max_features=2),
                      DecisionTreeClassifier(max_features= 10)]}
gridS=GridSearchCV(AdaBoostClassifier(base_estimator=DecisionTreeClassifier()),p)

#To provide the data and to fit the model
gridS.fit(X_train,y_train)
print(gridS.best_estimator_)


# In[50]:


#After parameter tuning we get the following
from sklearn.ensemble import AdaBoostClassifier

adaboost= AdaBoostClassifier(algorithm='SAMME.R',
                   base_estimator=DecisionTreeClassifier(ccp_alpha=0.0,
                                                         class_weight=None,
                                                         criterion='gini',
                                                         max_depth=1,
                                                         max_features=2,
                                                         max_leaf_nodes=None,
                                                         min_impurity_decrease=0.0,
                                                         min_impurity_split=None,
                                                         min_samples_leaf=1,
                                                         min_samples_split=2,
                                                         min_weight_fraction_leaf=0.0,
                                                         presort='deprecated',
                                                         random_state=None,
                                                         splitter='best'),
                   learning_rate=1.0, n_estimators=50, random_state=None)
                                                                   
#To provide the data and to fit the model
adaboost = adaboost.fit(X_train,y_train)

# to check the cross_val_score
scores = cross_val_score(adaboost, X_train, y_train, scoring="accuracy", cv=10)

print ("Cross Validated Accuracy: %0.5f +/- %0.5f" % (scores.mean(), scores.std()))


# In[51]:


#naming Features to be ranked
features = np.array(['alpha','b_mkt','b_smb','b_hml','tvol'])
features


# In[52]:


#Ranking features using Logistic Regression 
model = LogisticRegression()
# fit the model
model.fit(X_train, y_train)
# get importance
importance = model.coef_[0]
# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()


# In[54]:


#Trend Curve of the predicted stock for 2020
plt.plot(y_pred[:45])
plt.xlabel('Time')
plt.ylabel('Trend')
plt.title('Predicted Trend')
plt.show()


# For the DUK stocks I would suggest the client to hold the stocks until further event and if the stock goes does sell the stock for year 2020.If the client is new I would tell to buy the stocks after 30 days as per the model the stocks go down after 30 days

# In[ ]:




