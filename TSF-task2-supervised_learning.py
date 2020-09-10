#!/usr/bin/env python
# coding: utf-8

# # Task 2
# 
# # Supervised Learning
#   
#   In this section we are using Linear regression technique to predict the marks of the student based on the hours of  his/her     study

# In[51]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')


# In[52]:


#taking data
url = "http://bit.ly/w-data"
data = pd.read_csv(url)
data.fillna(-9999,inplace=True)
data
    


# In[53]:


#plotting to get the idea of dataset
data.plot(kind = 'scatter', x='Hours' , y='Scores', color='red')
plt.title('Study Hours VS Marks')
plt.xlabel('Study hours')
plt.ylabel('Marks')
plt.show()


# In[54]:


#data preparation
X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values  


# In[55]:


X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.4, random_state=0) 


# In[56]:


#applying regression and training the model
reg = LinearRegression()
model=reg.fit(X_train, y_train)


# In[57]:


reg.coef_


# In[58]:


reg.intercept_


# # Plotting regression line 

# In[59]:


#plotting regression line
reg_line = reg.coef_*X +reg.intercept_
plt.scatter(X,y)
plt.plot(X,reg_line, color='red')
plt.show()


# # Predicting some values 

# In[60]:


#predicting some values
x=[2.5,4,7]
x=pd.DataFrame(x)
y=model.predict(x)
y=pd.DataFrame(y)
df=pd.concat([x,y], axis=1,keys=['hours studied','predicted marks(%)'])
df


# # Final Prediction for 9.25 hrs

# In[61]:


#predicting marks for 9.25 hrs as per given in task 2
hours=[9.25]
hours=pd.DataFrame(hours)
marks=model.predict(hours)
marks=pd.DataFrame(marks)
data_frame=pd.concat([hours,marks], axis=1,keys=['hours studied','predicted marks(%)'])
data_frame


# In[62]:


# evaluation
y_predicted = model.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_predicted)) 

