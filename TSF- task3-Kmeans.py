#!/usr/bin/env python
# coding: utf-8

# # TSF-Task 3
# 
# ## To Explore Unsupervised Machine Learning
# 
# We are performing task 3 of unsupervised learning by predicting the optimal number of clusters by using K-Means Clustering Algorithm
# 

# In[1]:


#importing libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn import datasets


# In[2]:


#loading datasets
iris_data = datasets.load_iris()
iris_data


# In[3]:


df = pd.DataFrame(iris_data.data, columns = iris_data.feature_names)
df


# In[4]:


mms=MinMaxScaler()
mms.fit(df)
data_transformed=mms.transform(df)


# In[5]:


# Finding optimal value of K
Sum_of_Squared_distance=[]
K=range(1,15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(data_transformed)
    Sum_of_Squared_distance.append(km.inertia_)


# In[6]:


#using elbow method to predict K
plt.plot(K,Sum_of_Squared_distance,'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_Squared_distance')
plt.title('Elbow method for Optimal K')
plt.show()


# As we can see the elbow in the graph , that is our optimal value for K
# ,so from the graph we can see the optimal value is "3" for K

# In[7]:


# Applying K Means algorithm
km3 = KMeans(n_clusters=3)
km3 = km3.fit(df)


# In[8]:


print(km3.labels_)


# In[9]:


#cluster information
result = km3.labels_
result = pd.DataFrame(result, columns =['cluster'])
result.groupby('cluster').size()


# In[10]:


x = df.iloc[:,[0,1,2,3]].values
y = km3.fit_predict(x)


# In[11]:


# visualising the clusters by using first two columns
plt.scatter(x[y == 0, 0], x[y == 0, 1], 
            s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(x[y == 1, 0], x[y == 1, 1], 
            s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(x[y == 2, 0], x[y == 2, 1], 
            s = 100, c = 'black', label = 'Iris-virginica')

# Plotting the centroids of the clusters
plt.scatter(km3.cluster_centers_[:, 0], km3.cluster_centers_[:, 1],
            s = 100, c = 'yellow', label = 'Centroids')

plt.legend()


# In[ ]:




