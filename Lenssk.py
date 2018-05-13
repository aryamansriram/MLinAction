
# coding: utf-8

# In[1]:

import os
os.getcwd()


# In[5]:

os.chdir('C:\\Users\\Aryaman Sriram\Documents\machine_learning-master\Ch03')
    


# In[6]:

os.getcwd()


# In[12]:

import pandas as pd
DS=pd.read_csv('lenses.txt',sep="\t")


# In[28]:

import numpy as np
Y=DS.iloc[:,4]
X=DS.iloc[:,:-1]
X


# In[62]:

from sklearn.preprocessing import OneHotEncoder
enc=OneHotEncoder()
enc.fit_transform(X_train)
enc2=OneHotEncoder()
enc2.fit_transform(X_test)




# In[61]:

from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.33)



# In[84]:

from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier(min_samples_split=4)
clf.fit(X_train,Y_train)
pred=clf.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(pred,Y_test))


# In[ ]:



