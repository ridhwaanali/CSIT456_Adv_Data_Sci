#!/usr/bin/env python
# coding: utf-8

# In[28]:


#COVID-19 IMPACT ON STOCKS


# In[29]:


import pandas as pd
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(font_scale=1.2)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[30]:


stock =  pd.read_csv('stock_value.csv')
stock


# In[31]:


import matplotlib.pyplot as plt

Month = ['November', 'December', 'January', 'February', 'March', 'April']
IndexValue = [28051.41, 28538.44, 28256.03, 25409.36, 21917.16, 24345.72]
plt.title('COVID-19 IMPACT ON STOCKS')
plt.plot(Month,IndexValue)
plt.scatter(Month,IndexValue)
plt.xlabel('Month')
plt.ylabel('Index Value')


# In[32]:


import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

objects = ('November', 'December', 'January', 'February', 'March', 'April')
y_pos = np.arange(len(objects))
performance = [28051.41, 28538.44, 28256.03, 25409.36, 21917.16, 24345.72]
plt.bar(y_pos,performance,align='center', alpha=0.5, color=['yellow','green','pink', 'cyan','red','violet'])
plt.xticks(y_pos, objects)
plt.xlabel('Month')
plt.ylabel('Index Value')
plt.title('COVID-19 IMPACT ON STOCKS')
plt.show()


# In[33]:


from sklearn.model_selection import StratifiedKFold
skfold = StratifiedKFold(n_splits=5, shuffle=True)


# In[37]:


from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(Lasso(), param_grid, cv=10, return_train_score=True)


# In[ ]:




