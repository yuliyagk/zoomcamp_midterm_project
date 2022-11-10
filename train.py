#!/usr/bin/env python
# coding: utf-8

# # Midterm Project
# 
# The dataset which was used is at https://www.kaggle.com/datasets/deepu1109/star-dataset?resource=download

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import bentoml

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import xgboost as xgb


# In[2]:


data = pd.read_csv('./6_class_csv.csv')


# ### Star types
# 
# Brown Dwarf -> Star Type = 0
# 
# Red Dwarf -> Star Type = 1
# 
# White Dwarf-> Star Type = 2
# 
# Main Sequence -> Star Type = 3
# 
# Supergiant -> Star Type = 4
# 
# Hypergiant -> Star Type = 5
# 

# In[3]:


data.head()


# For easier access to the colums we remove the units and convert ' ' to underscore

# In[4]:


data.columns = data.columns.str.lower().str.replace(' ', '_')

units = ['_\(k\)', '\(l/lo\)', '\(r/ro\)', '\(mv\)']

for unit in units:
   data.columns = data.columns.str.lower().str.replace(unit, '')


# In[5]:


data.columns


# In[6]:


data.describe


# In[7]:


data.dtypes


# In[8]:


data.notnull().count()


# In[9]:


data.star_color.unique()


# It looks like that we need to unify the different color spellings

# In[10]:


replace_color = {
        'Blue ': 'blue',
        'Blue': 'blue',
        'Blue white': 'blue-white',
        'Blue white ': 'blue-white',
        'Blue-white': 'blue-white',
        'Blue White ': 'blue-white',
        'Blue White': 'blue-white',
        'Blue-White': 'blue-white',
        'yellow-white': 'yellow-white',
        'White-Yellow': 'yellow-white',
        'yellowish': 'yellowish',
        'Yellowish': 'yellowish',
        'Pale yellow orange': 'pale-yellow-orange',
        'Orange-Red': 'orange-red',
        'Red': 'red',
        'White': 'white',
        'Whitish': 'whitish',
        'Yellowish White': 'yellowish-white',
        'Orange': 'orange'
    }
data.star_color = data.star_color.replace(replace_color)


# In[11]:


data.star_color.unique()


# In[12]:


# make the visualitions for the better understanding of the classes
#plt.rcParams['figure.figsize'] = (15,8)
#sns.set_context("paper")

#sns.countplot(x='star_type',data=data)


# It seems that the dataset was already preprocessed and has balanced data.

# In[13]:


#df = pd.DataFrame(data)

#pd.plotting.scatter_matrix(df, alpha=0.2)


# # Train the dataset

# In[14]:


df_train, df_test = train_test_split(data, test_size=0.2, random_state=11)

df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.star_type
y_test = df_test.star_type

del df_train['star_type']
del df_test['star_type']


# In[15]:


y_train.unique()


# In[16]:


dv = DictVectorizer(sparse=False)

train_dicts = df_train.fillna(0).to_dict(orient='records')
X_train = dv.fit_transform(train_dicts)

test_dicts = df_test.fillna(0).to_dict(orient='records')
X_test = dv.transform(test_dicts)


# In[17]:


rf = RandomForestClassifier(n_estimators=200,
                            max_depth=10,
                            min_samples_leaf=3,
                            random_state=1)
rf.fit(X_train, y_train)


# In[18]:


y_pred = rf.predict(X_test)


# In[19]:


#disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred))
#disp.plot()
#plt.show()


# # Export the random forrest classifier for BentoML

# In[20]:


bentoml.sklearn.save_model(
    'star_type_model_skl',
    rf,
    custom_objects={
        'dictVectorizer': dv
    },
    signatures={ 
       'predict': {
           'batchable': True,
           'batch_dim': 0
           
       }}
)


# # Create model with XGBoost
# 
# For XGBoost we have to modify the y vector to map our values [0, 1, 2, 3, 4, 5] to values beween [0,1] because XGBoost expects that.

# In[21]:


replace_values = { 0: 0.0, 1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8, 5: 1.0}
y_train = y_train.replace(replace_values)


# In[22]:


y_train.unique()


# In[23]:


dtrain = xgb.DMatrix(X_train, label=y_train)


# In[24]:


xgb_params = {
    'eta': 0.1, 
    'max_depth': 3,
    'min_child_weight': 1,

    'objective': 'binary:logistic',
    'eval_metric': 'auc',

    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}

model = xgb.train(xgb_params, dtrain, num_boost_round=175)


# # Export the xgboost model using BentoML

# In[25]:


bentoml.xgboost.save_model(
    'star_type_model',
    model,
    custom_objects={
        'dictVectorizer': dv
    },
    signatures={ 
       'predict': {
           'batchable': True,
           'batch_dim': 0
           
       }}
)


# In[26]:


import json


# In[27]:


request = df_test.iloc[0].to_dict()
print(json.dumps(request, indent=2))


# In[ ]:




