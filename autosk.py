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
import json 
import time
from sklearn import preprocessing 


warnings.filterwarnings('ignore')


# In[41]:


def run_as(X, y, target_ft, time_budget=30, include_preprocessors = None, n_jobs=-1):
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
        per_run_time_limit=time_budget//10,
        tmp_folder='./tmp/autosklearn_regression_example_tmp',
        output_folder='./tmp/autosklearn_regression_example_out',
        include_preprocessors=include_preprocessors,
        ml_memory_limit=None,
        ensemble_memory_limit=3000,
        metric=autosklearn.metrics.f1_weighted,
        n_jobs=n_jobs
    )
    automl.fit(X_train, y_train)
    y_hat = automl.predict(X_test)
    
    acc = sklearn.metrics.accuracy_score(y_test, y_hat)
    f1_s = sklearn.metrics.f1_score(y_test, y_hat, average='weighted')
    metrs = []
    metrs.append("Accuracy score - " + str(acc))
    metrs.append("F1 score - " + str(f1_s))
    
    res = ["","","","",f1_s,acc,"",automl.show_models()]
    
    
    return str(metrs),res

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
    feature_matrix, feature_defs = ft.dfs(entityset = es, target_entity = 'data', verbose = 1)
                                      #agg_primitives=["mean", "max", "min", "std", "skew"],
                                     # trans_primitives = ['add_numeric', 'multiply_numeric'])
    return feature_matrix

def gen_feats_autofeat(X,y):
    fsel = autofeat.FeatureSelector(verbose=1)
    X = fsel.fit_transform(X,y)
    return X
    
    
def run_test(df_path,target_ft, mode = 0, time_budget=30,n_jobs=-1):
    results = []
    df = pd.read_csv(df_path)
    object_columns = df.select_dtypes(include='object')
    if len(object_columns.columns):
        df[object_columns.columns] = object_columns.apply(preprocessing.LabelEncoder().fit_transform)
    X = df.drop(columns=target_ft)
    y = df[target_ft]
    res_df = []
    autofeat_time = 0
    
    if mode == 0 or mode == 1:
        start = time.monotonic()
        X_new = gen_feats_autofeat(X,y)
        end = time.monotonic()
        autofeat_time = int(end-start)
        print("Autofeat_time: ",autofeat_time)
        rs = run_as(X_new,y,target_ft,time_budget=time_budget, include_preprocessors =["no_preprocessing"],n_jobs=n_jobs)
        results.append("Autosk Only with Preprocessing: " + rs[0])
        rs[1][0] = df_path[5:-4]
        rs[1][1] = str(time_budget/60)+'m'
        rs[1][2] = "Autofeat"
        rs[1][3] = "AutoSK"
        rs[1][6] = str(X_new.shape)
        res_df.append(rs[1])
    

    if mode == 0 or mode == 2:
        rs = run_as(X,y,target_ft, time_budget=time_budget+autofeat_time, include_preprocessors=None,n_jobs=n_jobs)   
        results.append("Autosk Only with Preprocessing: " + rs[0])
        rs[1][0] = df_path[5:-4]
        rs[1][1] = str(round((time_budget+autofeat_time)/60,2))+'m'
        rs[1][2] = "AutoSK"
        rs[1][3] = "AutoSK"
        rs[1][6] = str(X.shape)
        res_df.append(rs[1])
    if mode == 0 or mode == 3:
        rs = run_as(X,y,target_ft,time_budget=time_budget, include_preprocessors =["no_preprocessing"],n_jobs=n_jobs)
        results.append("Autosk Only without Preprocessing: " + rs[0])
        rs[1][0] = df_path[5:-4]
        rs[1][1] = str(round((time_budget+autofeat_time)/60,2))+'m'
        rs[1][2] = "None"
        rs[1][3] = "AutoSK"
        rs[1][6] = str(X.shape)
        res_df.append(rs[1])
    
    
    
    
    #if mode == 0 or mode == 4:
    #    X_new = gen_feats_featools(X)
    #    rs = run_as(X_new,y,target_ft,time_budget=time_budget, include_preprocessors =["no_preprocessing"])
    #    results.append("Autosk with Featuretools: " + rs)

    #if mode ==0 or mode == 5:
    #    rs = run_tpot(X,y,target_ft, time_budget=time_budget, include_preprocessors=None)   
    #    results.append("TPOT Only with Preprocessing: " + rs)
    
    
        
    print("===================================")
    print("Time budeget: ",time_budget)
    [print(x) for x in results]
    
    res_df =  pd.DataFrame(res_df, columns = ["Dataset","Time","Preprocessing","AutoML","Accuracy","F1","Shape","PipeLine"])
    res_df.drop(columns=["PipeLine"]).to_csv("results/"+df_path[5:])
    res_df.to_csv("results/pipe_"+df_path[5:])
    
    return res_df


# In[47]:


#get_ipython().system('rm -r tmp')
#df_path = "data/gina.csv"
#target_ft = "class"
#res = run_test(df_path, target_ft, mode=2 ,n_jobs=-1,time_budget=120)


# In[38]:


#!rm -r tmp
#df = pd.read_csv("blood.csv")
#target_ft = "class"
#run_test(df, target_ft, mode=2, time_budget=60)


# In[7]:


#df = pd.read_csv("winequality-red.csv")
#target_ft = "quality"
#run_test(df, target_ft, mode=2 ,time_budget=30)


# In[6]:


#df = pd.read_csv("data/airlines.csv")


# In[7]:


#!rm -r tmp
#df = pd.read_csv("data/airlines.csv").drop(columns=["Airline","AirportFrom","AirportTo"])
#target_ft = "Delay"
#run_test(df, target_ft, mode=2 ,time_budget=30)


# In[50]:


#!rm -r tmp
#df = "data/gina.csv"
#target_ft = "class"
#run_test(df, target_ft, mode=2 ,time_budget=30)


# In[9]:


#!rm -r tmp
#df = pd.read_csv("data/gina.csv")
#target_ft = "class"
#run_test(df, target_ft, mode=4 ,time_budget=420)


# In[16]:


#!ls data


# In[49]:


#!rm -r tmp
#df_path = "data/20_newsgroups.csv"
#target_ft = "class"
#res = run_test(df_path, target_ft, mode=2 ,time_budget=3600, n_jobs=1)


# In[ ]:

#df_path = "data/rcv1.csv"
df_path = "data/gina.csv"
target_ft = "class"
res = run_test(df_path, target_ft, mode=0 ,time_budget=3600, n_jobs=1)



