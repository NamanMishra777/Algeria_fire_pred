#!/usr/bin/env python
# coding: utf-8

# In[44]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[45]:


dataset = pd.read_csv("C:\\Users\\naman\\OneDrive\\Desktop\\Ai&ML_proj\\Algerian_forest_fires_cleaned_dataset.csv")
dataset


# In[46]:


dataset.info()


# In[47]:


dataset.describe()


# In[48]:


dataset.head()


# In[49]:


dataset.tail()


# In[50]:


dataset['Classes'].value_counts()


# In[51]:


dataset['Classes'] = np.where(dataset['Classes'].str.contains("not fire"),0,1) 


# In[52]:


dataset['Classes'].value_counts()


# In[53]:


dataset.tail()


# In[54]:


dataset.columns


# In[55]:


dataset.drop(['day','month','year'],axis=1,inplace=True)


# In[56]:


dataset.head()


# In[57]:


X = dataset.drop('FWI',axis=1)
y = dataset['FWI']


# In[58]:


X.head()


# In[59]:


y


# In[60]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.20,random_state=44)


# In[61]:


X_train.shape,X_test.shape


# In[62]:


X_train.corr()


# In[63]:


plt.figure(figsize=(12,10))
corr=X_train.corr()
sns.heatmap(corr,annot=True)


# In[64]:


X_train.corr()


# In[65]:


def correlation(dataset, threshold):
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: 
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr


# In[66]:


corr_features=correlation(X_train,0.85)
corr_features


# In[67]:


## drop features when correlation is more than 0.85 
X_train.drop(corr_features,axis=1,inplace=True)
X_test.drop(corr_features,axis=1,inplace=True)
X_train.shape,X_test.shape


# In[68]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)


# In[71]:


X_train_scaled


# In[72]:


plt.subplots(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.boxplot(data=X_train)
plt.title('Before Scaling')
plt.subplot(1, 2, 2)
sns.boxplot(data=X_train_scaled)
plt.title('After Scaling')


# In[73]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
linreg=LinearRegression()
linreg.fit(X_train_scaled,y_train)
y_pred=linreg.predict(X_test_scaled)
m=mean_absolute_error(y_test,y_pred)
score=r2_score(y_test,y_pred)
print("Mean absolute error", m)
print("R2 Score", score)
plt.scatter(y_test,y_pred)


# In[74]:


from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
lasso=Lasso()
lasso.fit(X_train_scaled,y_train)
y_pred=lasso.predict(X_test_scaled)
m=mean_absolute_error(y_test,y_pred)
score=r2_score(y_test,y_pred)
print("Mean absolute error", m)
print("R2 Score", score)
plt.scatter(y_test,y_pred)


# ### Cross Validation Lasso

# In[75]:


from sklearn.linear_model import LassoCV
lassocv=LassoCV(cv=5)
lassocv.fit(X_train_scaled,y_train)


# In[76]:


y_pred=lassocv.predict(X_test_scaled)
plt.scatter(y_test,y_pred)
m=mean_absolute_error(y_test,y_pred)
score=r2_score(y_test,y_pred)
print("Mean absolute error", m)
print("R2 Score", score)


# Ridge Regression

# In[77]:


from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
ridge=Ridge()
ridge.fit(X_train_scaled,y_train)
y_pred=ridge.predict(X_test_scaled)
m=mean_absolute_error(y_test,y_pred)
score=r2_score(y_test,y_pred)
print("Mean absolute error", m)
print("R2 Score", score)
plt.scatter(y_test,y_pred)


# In[78]:


from sklearn.linear_model import RidgeCV
ridgecv=RidgeCV(cv=5)
ridgecv.fit(X_train_scaled,y_train)
y_pred=ridgecv.predict(X_test_scaled)
plt.scatter(y_test,y_pred)
m=mean_absolute_error(y_test,y_pred)
score=r2_score(y_test,y_pred)
print("Mean absolute error", m)
print("R2 Score", score)


# In[79]:


ridgecv.get_params()


# In[80]:


from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
elastic=ElasticNet()
elastic.fit(X_train_scaled,y_train)
y_pred=elastic.predict(X_test_scaled)
m=mean_absolute_error(y_test,y_pred)
score=r2_score(y_test,y_pred)
print("Mean absolute error", m)
print("R2 Score", score)
plt.scatter(y_test,y_pred)

