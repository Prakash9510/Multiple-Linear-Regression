#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing the libraries


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


# Importing the dataset


# In[4]:


Dataset = pd.read_csv("50_Startups.csv")
Dataset.head()


# In[5]:


X = Dataset.iloc[:,:-1]
Y = Dataset.iloc[:,-1]


# In[7]:


# convert the categorical column into continous column

states = pd.get_dummies(X['State'],drop_first=True)


# In[8]:


# Drop the state column
X=X.drop('State',axis=1)


# In[9]:


#concat the dummy variable
X = pd.concat([X,states],axis=1)


# In[10]:


X.head()


# In[11]:


# Splitting the dataset into the training and test set

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2, random_state=20)


# In[12]:


# Fitting multiple linear regression to the training set

from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X_train,Y_train)


# In[13]:


# Predicting the test set results
y_pred = LR.predict(X_test)


# In[15]:


from sklearn.metrics import r2_score
score = r2_score(Y_test,y_pred)


# In[17]:


score


# In[ ]:




