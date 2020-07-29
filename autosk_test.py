#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
import pandas as pd
import autosklearn.classification
import os
import featuretools as ft
import warnings
import autofeat 
from tpot import TPOTClassifier

warnings.filterwarnings('ignore')


# In[2]:


def run_as(X, y, target_ft, time_budget=30, include_preprocessors = None):
    try:
        os.remove('/tmp/autosklearn_regression_example_tmp')
        os.remove('/tmp/autosklearn_regression_example_out')
    except:
        pass
    #X = df.drop(columns=target_ft)
    #y = df[target_ft]
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=1)
    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=time_budget,
        per_run_time_limit=30,
        tmp_folder='./tmp/autosklearn_regression_example_tmp',
        output_folder='./tmp/autosklearn_regression_example_out',
        include_preprocessors = include_preprocessors
    )
    automl.fit(X_train, y_train)
    y_hat = automl.predict(X_test)
    
    metrs = []
    metrs.append("Accuracy score - " + str(sklearn.metrics.accuracy_score(y_test, y_hat)))
    metrs.append("F1 score - " + str(sklearn.metrics.f1_score(y_test, y_hat, average='macro')))
    
    print(automl.show_models())
    
    return str(metrs)

def run_tpot(X,y, target_ft,time_budget=30, include_preprocessors=None ):
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=1)
    pipeline_optimizer = TPOTClassifier(max_time_mins = time_budget/60, generations=None)
    pipeline_optimizer.fit(X_train, y_train)
    y_hat = pipeline_optimizer.predict(X_test)
    metrs = []
    metrs.append("Accuracy score - " + str(sklearn.metrics.accuracy_score(y_test, y_hat)))
    metrs.append("F1 score - " + str(sklearn.metrics.f1_score(y_test, y_hat, average='macro')))
    return str(metrs)

    
def gen_feats_featools(df):
    es = ft.EntitySet(id = 'df')
    es.entity_from_dataframe(entity_id = 'data', dataframe = df, 
                         make_index = True, index = 'index')
    feature_matrix, feature_defs = ft.dfs(entityset = es, target_entity = 'data',
                                      agg_primitives=["mean", "max", "min", "std", "skew"],
                                      trans_primitives = ['add_numeric', 'multiply_numeric'])
    return feature_matrix

def gen_feats_autofeat(X,y):
    fsel = autofeat.FeatureSelector(verbose=1)
    X = fsel.fit_transform(X,y)
    return X
    
    
def run_test(df,target_ft, mode = 0, time_budget=30):
    results = []
    X = df.drop(columns=target_ft)
    y = df[target_ft]
    if mode ==0 or mode == 1:
        rs = run_tpot(X,y,target_ft, time_budget=time_budget, include_preprocessors=None)   
        results.append("TPOT Only with Preprocessing: " + rs)
    if mode == 0 or mode == 2:
        rs = run_as(X,y,target_ft, time_budget=time_budget, include_preprocessors=None)   
        results.append("Autosk Only with Preprocessing: " + rs)
    if mode == 0 or mode == 3:
        rs = run_as(X,y,target_ft,time_budget=time_budget, include_preprocessors =["no_preprocessing"])
        results.append("Autosk Only without Preprocessing: " + rs)
   # if mode == 0 or mode == 4:
   #     X_new = gen_feats_featools(X)
   #     rs = run_as(X_new,y,target_ft,time_budget=time_budget, include_preprocessors =["no_preprocessing"])
   #     results.append("Autosk with Featuretools: " + rs)
    if mode == 0 or mode == 5:
        X_new = gen_feats_autofeat(X,y)
        rs = run_as(X_new,y,target_ft,time_budget=time_budget, include_preprocessors =["no_preprocessing"])
        results.append("Autosk with Autofeat: " + rs)
    
    
    print("===================================")
    print("Time budeget: ",time_budget)
    [print(x) for x in results]



# In[3]:


#get_ipython().system('rm -r tmp')
#df = pd.read_csv("data/gina.csv")
df = pd.read_csv("data/ailerons_fx.csv")
target_ft = "binaryClass"
run_test(df, target_ft, mode=2 ,time_budget=30)


# In[ ]:




