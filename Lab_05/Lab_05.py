#!/usr/bin/env python
# coding: utf-8

# # Lab05 - Advanced Data Analytics and Machine Learning

# In[ ]:


#Download the dataset credit-data-post-import.csv, randomly split your dataset in two datasets: training (75% observations) and testing (25% observations). We'll use the training set to calibrate our model and then use the test set to evaluate how effective it is.
#Split our training data into 2 groups; data containing nulls and data not containing nulls on the monthly_income variable. Train on the latter and make 'predictions' on the null data to impute monthly_income using a regression algorithm with the variables 'number_real_estate_loans_or_lines' and 'number_of_open_credit_lines_and_loans'. 
#Save your train and test datasets in the csv files: credit-data-trainingset.csv and credit-data-testset.csv.
#For this first part, the report can simply compare the number of observations of each dataset (train and test) before and after the correction of null values.


# In[ ]:


import os
import numpy as np
import pandas as pd
import math
import matplotlib.pylab as plt
import seaborn as sns
from sklearn import linear_model
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = 15, 12


get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style='ticks', palette='Set2')


# In[ ]:


path="C:\Kavit\Stevens Institute of Technology\SEM 3\BIA 656 Advanced Data Analytics and Machine Learning\Assignment\Lab05\credit-data-post-import.csv"
df = pd.read_csv(path)
df.head(10)


# In[ ]:


len(df)


# In[ ]:


X = df[['revolving_utilization_of_unsecured_lines', 'age', 'number_of_time30-59_days_past_due_not_worse', 'debt_ratio', 'monthly_income', 'number_of_open_credit_lines_and_loans', 'number_of_times90_days_late', 'number_real_estate_loans_or_lines', 'number_of_time60-89_days_past_due_not_worse', 'number_of_dependents']]
y = df['monthly_income']

train_n, test_n = train_test_split(df, test_size=0.25, random_state=30)
print(len(train_n))
print(len(test_n))


# In[ ]:


nn= train_n[train_n['monthly_income'].notnull()]
nn.head()
len(nn)


# In[ ]:


nulls = train_n[train_n['monthly_income'].isnull()]
nulls.head()
len(nulls)


# In[ ]:


X = nn[['number_real_estate_loans_or_lines', 'number_of_open_credit_lines_and_loans']]
y = nn['monthly_income']
reg = LinearRegression().fit(X, y)


# In[ ]:


len(X)


# In[ ]:


predict = nulls.drop(['monthly_income'], axis=1)
predict = predict[['number_real_estate_loans_or_lines', 'number_of_open_credit_lines_and_loans']]
preds = reg.predict(predict)
preds


# In[ ]:


nulls['monthly_income'] = preds

nulls.head()


# In[ ]:


#final train dataset 

train = nn.append(nulls)
print(len(train))
train.head(20)


# In[ ]:


#Converting the final train dataset to CSV file
train.to_csv(r'C:\Kavit\Stevens Institute of Technology\SEM 3\BIA 656 Advanced Data Analytics and Machine Learning\Assignment\Lab05\credit-data-trainingset.csv',index=False)


# In[ ]:


nulls = test_n[test_n['monthly_income'].isnull()]
not_null = test_n[test_n['monthly_income'].notnull()]
predict1 = nulls.drop(['monthly_income'], axis=1)
predict1 = predict1[['number_real_estate_loans_or_lines', 'number_of_open_credit_lines_and_loans']]
preds = reg.predict(predict1)
preds


# In[ ]:


len(nulls)


# In[ ]:


nulls['monthly_income'] = preds

nulls.head()


# In[ ]:


#final test dataset 

test = not_null.append(nulls)
print(len(test))
test.head(20)


# In[ ]:


#Converting the final test dataset to CSV file

test.to_csv(r'C:\Kavit\Stevens Institute of Technology\SEM 3\BIA 656 Advanced Data Analytics and Machine Learning\Assignment\Lab05\credit-data-testset.csv',index=False)


# In[ ]:


#Training dataset with nulls
train_n.isnull().sum()


# In[ ]:


#Training dataset with imputed values
train.isnull().sum()


# In[ ]:


#testing dataset with nulls
test_n.isnull().sum()


# In[ ]:


#Testing dataset with imputed values
test.isnull().sum()


# # Part - 2

# In[ ]:


path="C:\Kavit\Stevens Institute of Technology\SEM 3\BIA 656 Advanced Data Analytics and Machine Learning\Assignment\Lab05\credit-data-trainingset.csv"
train = pd.read_csv(path)
train.head(5)


# In[ ]:


#Loading the Test Data
path="C:\Kavit\Stevens Institute of Technology\SEM 3\BIA 656 Advanced Data Analytics and Machine Learning\Assignment\Lab05\credit-data-trainingset.csv"
test = pd.read_csv(path)
test.head(10)

X_test=test.drop(['serious_dlqin2yrs'],axis=1)
y_test=test['serious_dlqin2yrs']


# # Logistic Regression with penalty = 'l2'

# In[ ]:


X_train=train.drop(['serious_dlqin2yrs'],axis=1)
y_train=train['serious_dlqin2yrs']
sd=LogisticRegression(penalty='l2').fit(X_train, y_train)


# In[ ]:


from sklearn.model_selection import cross_val_score

scores = cross_val_score(sd, X_train, y_train, scoring="accuracy", cv=10)

print ("Cross Validated Accuracy: %0.5f +/- %0.5f" % (scores.mean(), scores.std()))

#Cross Validated Accuracy: 0.93275 +/- 0.00021


# ### ROC Curve for Logistic Regression 

# In[ ]:


#Get the probability of Y_test records being = 1
Y_test_probability_1 = sd.predict_proba(X_test)[:, 1]

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


# ### Cumulative  

# In[ ]:


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
plt.plot(x_cumulative, y_cumulative,label=sd)


# ### Lift Curve 

# In[ ]:


#Plotting the Lift Curve

plt.plot(x_cumulative, y_cumulative/x_cumulative, label=sd)
plt.plot([0,100], [1,1], 'k--', label="Random")
plt.xlabel("Percentage of test instances (decreasing score)")
plt.ylabel("Lift (times)")
plt.title("Lift curve")
plt.legend()


# ## Decisiton tree 

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier(criterion='entropy',splitter='best',max_depth=2)

#To provide the data and to fit the model
d_t=decision_tree.fit(X_train,y_train)

# to check the cross_val_score
scores = cross_val_score(d_t, X_train, y_train, scoring="accuracy", cv=10)

print ("Cross Validated Accuracy: %0.5f +/- %0.5f" % (scores.mean(), scores.std()))


# ### ROC Curve 

# In[ ]:


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


# ### Cumulative 

# In[ ]:


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


# ### Lift Curve 

# In[ ]:


#Plotting the curve

plt.plot(x_cumulative, y_cumulative/x_cumulative, label=d_t)
plt.plot([0,100], [1,1], 'k--', label="Random")
plt.xlabel("Percentage of test instances (decreasing score)")
plt.ylabel("Lift (times)")
plt.title("Lift curve")
plt.legend()


# ## Support Vector Machine (SVM) Classifier 

# In[ ]:


from sklearn.svm import LinearSVC


svmc = LinearSVC(penalty='l2', loss='squared_hinge')

#To provide the data and to fit the model
svmc =svmc.fit(X_train,y_train)

# to check the cross_val_score
scores = cross_val_score(svmc, X_train, y_train, scoring="accuracy", cv=10)

print ("Cross Validated Accuracy: %0.5f +/- %0.5f" % (scores.mean(), scores.std()))


# ### ROC Curve

# In[ ]:


#LinearSVC is used here for fast processing even though it misses the probability function, 
#but with CalibratedClassifierCV we can use it for probability functions which does the same work as svm.SVC's(predict_proba())
#function.
#visit site for more info: https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html and https://scikit-learn.org/stable/modules/calibration.html

from sklearn.calibration import CalibratedClassifierCV

clf = CalibratedClassifierCV(svmc) 
clf= clf.fit(X_train, y_train)
# Get the probability of Y_test records being = 1
Y_test_probability_1 = clf.predict_proba(X_test)[:, 1]

# Use the metrics.roc_curve function to get the true positive rate (tpr) and false positive rate (fpr)
fpr, tpr, thresholds = metrics.roc_curve(y_test, Y_test_probability_1)
    
# Get the area under the curve (AUC)
auc = scores.mean()

# Plot the ROC curve
plt.plot(fpr, tpr, label="AUC " + str(round(auc, 2)))
plt.xlabel("False positive rate (fpr)")
plt.ylabel("True positive rate (tpr)")
plt.plot([0,1], [0,1], 'k--', label="Random")
plt.legend(loc='best')


# ### Cumulative 

# In[ ]:


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
plt.plot(x_cumulative, y_cumulative,label=clf)


# ### Lift Curve 

# In[ ]:


#Plotting the graph

plt.plot(x_cumulative, y_cumulative/x_cumulative, label=clf)
plt.plot([0,100], [1,1], 'k--', label="Random")
plt.xlabel("Percentage of test instances (decreasing score)")
plt.ylabel("Lift (times)")
plt.title("Lift curve")
plt.legend()


# # Adaboost 

# In[ ]:


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

#Without parameter tuning
#Cross Validated Accuracy: 0.93500 +/- 0.00160


# ### ROC Curve

# In[ ]:


Y_test_probability_1 = adaboost.predict_proba(X_test)[:, 1]

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


# ### Cumulative 

# In[ ]:


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


# ### Lift Curve 

# In[ ]:


#Plotting the Lift Curve

plt.plot(x_cumulative, y_cumulative/x_cumulative, label=adaboost)
plt.plot([0,100], [1,1], 'k--', label="Random")
plt.xlabel("Percentage of test instances (decreasing score)")
plt.ylabel("Lift (times)")
plt.title("Lift curve")
plt.legend()


# # GridSearchSV

# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

#Since Adaboost had the best accuracy of all the models it has been selected for the GridSearchSV

p ={'base_estimator__max_depth':[1,100],
    'base_estimator':[DecisionTreeClassifier(max_features=2),
                      DecisionTreeClassifier(max_features= 10)]}
gridS=GridSearchCV(AdaBoostClassifier(base_estimator=DecisionTreeClassifier()),p)

#To provide the data and to fit the model
gridS.fit(X_train,y_train)
print(gridS.best_estimator_)


# # Testing the model with the best parameters 

# In[ ]:


from sklearn.metrics import accuracy_score

y_predict = adaboost.predict(X_test)
y_test = test['serious_dlqin2yrs']

accuracy_score(y_test,y_predict)

