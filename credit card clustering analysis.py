#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pandas')
get_ipython().system('pip install seaborn')
get_ipython().system('pip install numpy')
get_ipython().system('pip install matplotlib.pyplot')


# In[2]:


import pandas as pd
import numpy as np
clustering=pd.read_csv('datasets_14701_19663_CC GENERAL.csv')

clustering.head()
clustering.isnull().sum()
clustering['MINIMUM_PAYMENTS'].fillna(value=clustering['MINIMUM_PAYMENTS'].mean(),inplace=True)
clustering.isnull().sum()
clustering['CREDIT_LIMIT'].fillna(value=clustering['CREDIT_LIMIT'].mean(),inplace=True)

import seaborn as sns



transf=clustering.loc[:,'BALANCE':'TENURE']
#data has many outliers when the description is seen,hence we log_transform it,and the '0' value becomes -1 
# so we add 1 in all values 
log_transformed=np.log(transf+1)
log_transformed.describe()


# In[3]:



from sklearn.cluster import KMeans
#value is set to 30 and the elbow point is used to identify the best k
n_clusters=30
k_values=[]
for i in range(1,n_clusters):
    kmean= KMeans(i)
    kmean.fit(log_transformed)
    k_values.append(kmean.inertia_) 

import matplotlib.pyplot as plt
plt.plot(k_values, 'bx-')
# we noticed that after 4 the value doesnot change much
kmean= KMeans(n_clusters=4,init='k-means++',n_init=12)
kmean.fit(log_transformed)
labels=kmean.labels_
clusters=pd.concat([log_transformed, pd.DataFrame({'cluster':labels})], axis=1)
clusters.describe()


# In[4]:


for s in clusters:
    grids=sns.FacetGrid(clusters,col='cluster')
    grids.map(plt.hist,s)


# In[10]:


from sklearn.decomposition import PCA
X=log_transformed.loc[:,'BALANCE':'TENURE'].values
pca=PCA(n_components=2)
pca.fit(X)
X_pca=pca.transform(X)
X_pca.shape
x,y=X_pca[:,0],X_pca[:,1]
new_df=pd.DataFrame({'x':x,'y':y,'label':labels})
grouped=new_df.groupby('label')
grouped.head()
fig, ax = plt.subplots(figsize=(20, 13)) 
colors = {0: 'red',
          1: 'blue',
          2: 'green', 
          3: 'yellow', 
          4: 'orange',  
          5:'purple'}

names = {0: 'who make all type of purchases', 
         1: 'more people with due payments', 
         2: 'who purchases mostly in installments', 
         3: 'who take more cash in advance', 
         4: 'who make expensive purchases',
         5:'who don\'t spend much money'}
for name, group in grouped:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=5,
            color=colors[name],label=names[name], mec='none')
    ax.set_aspect('auto')
    ax.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
    ax.tick_params(axis= 'y',which='both',left='off',top='off',labelleft='off')
    
ax.legend()
ax.set_title("Customers Segmentation based on their Credit Card usage bhaviour.")
plt.show()


# In[11]:


clusters.groupby('cluster').mean


# In[ ]:




