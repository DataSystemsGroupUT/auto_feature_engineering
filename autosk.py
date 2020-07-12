#!/usr/bin/env python
# coding: utf-8

# In[5]:


import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
import pandas as pd
import autosklearn.classification
import os


# In[15]:


def run_as(df, target_ft):
    try:
        os.remove('/tmp/autosklearn_regression_example_tmp')
        os.remove('/tmp/autosklearn_regression_example_out')
    except:
        pass
    X = df.drop(columns=target_ft)
    y = df[target_ft]
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=1)
    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=30,
        per_run_time_limit=30,
        tmp_folder='./tmp/autosklearn_regression_example_tmp',
        output_folder='./tmp/autosklearn_regression_example_out',
    )
    automl.fit(X_train, y_train)
    y_hat = automl.predict(X_test)
    print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_hat))


# In[16]:


df = pd.read_csv("blood.csv")
target_ft = "class"


# In[ ]:


run_as(df,target_ft)


# In[ ]:




