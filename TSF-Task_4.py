#!/usr/bin/env python
# coding: utf-8

# # TSF - Task4
# 
# ### To Explore Decision Tree Algorithm
# 
# Here we are using decision tree algorithm on iris dataset
# 

# In[1]:


#import libraries
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


#importing dataset and tree
from sklearn.datasets import load_iris
from sklearn import tree


# In[5]:


#using decision tree classifier
clf = tree.DecisionTreeClassifier(random_state = 0)
iris = load_iris()
clf = clf.fit(iris.data, iris.target)


# In[7]:


iris


# In[8]:


iris.target


# ## visualising decision tree
# 

# In[6]:


plt.figure(figsize=(15,10))
tree.plot_tree(clf,filled=True)


# In[9]:


# text form
print(tree.export_text(clf))


# In[ ]:




