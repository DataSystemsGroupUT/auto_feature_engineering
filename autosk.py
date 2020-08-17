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
from datetime import datetime


warnings.filterwarnings('ignore')


# In[2]:


#from dask.distributed import Client
#client = Client(n_workers=6, threads_per_worker=1, memory_limit=None)


# In[3]:


#client


# In[4]:


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
        #per_run_time_limit=30,
        tmp_folder='./tmp/autosklearn_regression_example_tmp',
        output_folder='./tmp/autosklearn_regression_example_out',
        include_preprocessors=include_preprocessors,
        ml_memory_limit=None,
        ensemble_memory_limit=None,
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

def run_tpot(X,y, target_ft,time_budget=30, include_preprocessors=None, n_jobs=1 ):

    print(n_jobs)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=1)
    
    if include_preprocessors:
        pipeline_optimizer = TPOTClassifier(max_time_mins = time_budget//60, generations=None,
                                            use_dask=False,
                                            n_jobs=n_jobs,)
    else:
        pipeline_optimizer = TPOTClassifier(max_time_mins = time_budget//60, generations=None,
                                    use_dask=False,
                                    template='Classifier',
                                    n_jobs=n_jobs,)
    
    pipeline_optimizer.fit(X_train, y_train)
    y_hat = pipeline_optimizer.predict(X_test)
    acc = sklearn.metrics.accuracy_score(y_test, y_hat)
    f1_s = sklearn.metrics.f1_score(y_test, y_hat, average='weighted')
    metrs = []
    metrs.append("Accuracy score - " + str(acc))
    metrs.append("F1 score - " + str(f1_s))
    res = ["","","","",f1_s,acc,"",pipeline_optimizer.export()]
    
    
    return str(metrs),res

    
def gen_feats_featools(df):
    es = ft.EntitySet(id = 'df')
    es.entity_from_dataframe(entity_id = 'data', dataframe = df, 
                         make_index = True, index = 'index')
    feature_matrix, feature_defs = ft.dfs(entityset = es, target_entity = 'data', verbose = 1)
                                      #agg_primitives=["mean", "max", "min", "std", "skew"],
                                     # trans_primitives = ['add_numeric', 'multiply_numeric'])
    return feature_matrix

def gen_feats_autofeat(X,y,af_iters=5):
    fsel = autofeat.FeatureSelector(verbose=1,featsel_runs=af_iters)
    X = fsel.fit_transform(X,y)
    return X
    
    
def run_test(df_path,target_ft, mode = 0, time_budget=30,n_jobs=-1, af_iters=5):
    now = datetime.now()

    current_time = now.strftime("%H:%M:%S")
    print("Start Time =", current_time)
    print("Time budget =", time_budget//60)
    results = []
    df = pd.read_csv(df_path)
    object_columns = df.select_dtypes(include='object')
    if len(object_columns.columns):
        df[object_columns.columns] = object_columns.apply(preprocessing.LabelEncoder().fit_transform)
    X = df.drop(columns=target_ft)
    y = df[target_ft]
    res_df = []
    autofeat_time = 0
    
    #if mode == 0 or mode == 1:
    #    start = time.monotonic()
    #    X_new = gen_feats_autofeat(X,y)
    #    end = time.monotonic()
    #    autofeat_time = int(end-start)
    #    print("Autofeat_time: ",autofeat_time)
    #    rs = run_as(X_new,y,target_ft,time_budget=time_budget, include_preprocessors =["no_preprocessing"],n_jobs=n_jobs)
    #    results.append("Autosk Only with Preprocessing: " + rs[0])
    #    rs[1][0] = df_path[5:-4]
    #    rs[1][1] = str(time_budget/60)+'m'
    #    rs[1][2] = "Autofeat"
    #    rs[1][3] = "AutoSK"
    #    rs[1][6] = str(X_new.shape)
    #    res_df.append(rs[1])
    

    #if mode == 0 or mode == 2:
    #    rs = run_as(X,y,target_ft, time_budget=time_budget+autofeat_time, include_preprocessors=None,n_jobs=n_jobs)   
    #    results.append("Autosk Only with Preprocessing: " + rs[0])
    #    rs[1][0] = df_path[5:-4]
    #    rs[1][1] = str(round((time_budget+autofeat_time)/60,2))+'m'
    #    rs[1][2] = "AutoSK"
    #    rs[1][3] = "AutoSK"
    #    rs[1][6] = str(X.shape)
    #    res_df.append(rs[1])
    #if mode == 0 or mode == 3:
    #    rs = run_as(X,y,target_ft,time_budget=time_budget, include_preprocessors =["no_preprocessing"],n_jobs=n_jobs)
    #    results.append("Autosk Only without Preprocessing: " + rs[0])
    #    rs[1][0] = df_path[5:-4]
    #    rs[1][1] = str(round((time_budget+autofeat_time)/60,2))+'m'
    #    rs[1][2] = "None"
    #    rs[1][3] = "AutoSK"
    #    rs[1][6] = str(X.shape)
    #    res_df.append(rs[1])
    
    
    
    
    #if mode == 0 or mode == 4:
    #    X_new = gen_feats_featools(X)
    #    rs = run_as(X_new,y,target_ft,time_budget=time_budget, include_preprocessors =["no_preprocessing"])
    #    results.append("Autosk with Featuretools: " + rs)

    

    if mode ==0 or mode == 2:
        start = time.monotonic()
        X_new = gen_feats_autofeat(X,y,af_iters=af_iters)
        print("X old shape:", X.shape)
        print("X new shape:", X_new.shape)
        end = time.monotonic()
        autofeat_time = int(end-start)
        start = time.monotonic()
        rs = run_tpot(X_new,y,target_ft, time_budget=time_budget, include_preprocessors=None, n_jobs=n_jobs)   
        end = time.monotonic()
        print("Actual Time Taken: ",str(end-start))
        results.append("TPOT with Autofeat: " + rs[0])
        rs[1][0] = df_path[5:-4]
        rs[1][1] = str(round((time_budget+autofeat_time)/60,2))+'m'
        rs[1][2] = "AutoFeat"
        rs[1][3] = "TPOT"
        rs[1][6] = str(X_new.shape)
        res_df.append(rs[1])

    
    
    if mode ==0 or mode == 5:
        start = time.monotonic()
        rs = run_tpot(X,y,target_ft, time_budget=time_budget+autofeat_time, include_preprocessors=False, n_jobs=n_jobs)   
        end = time.monotonic()
        print("Actual Time Taken: ",str(end-start))
        results.append("TPOT Only with No Preprocessing: " + rs[0])
        rs[1][0] = df_path[5:-4]
        rs[1][1] = str(round((time_budget+autofeat_time)/60,2))+'m'
        rs[1][2] = "None"
        rs[1][3] = "TPOT"
        rs[1][6] = str(X.shape)
        res_df.append(rs[1])
        
        

    if mode ==0 or mode == 1:
        start = time.monotonic()
        rs = run_tpot(X,y,target_ft, time_budget=time_budget+autofeat_time, include_preprocessors=True, n_jobs=n_jobs)   
        end = time.monotonic()
        print("Actual Time Taken: ",str(end-start))
        results.append("TPOT Only with Preprocessing: " + rs[0])
        rs[1][0] = df_path[5:-4]
        rs[1][1] = str(round((time_budget+autofeat_time)/60,2))+'m'
        rs[1][2] = "TPOT"
        rs[1][3] = "TPOT"
        rs[1][6] = str(X.shape)
        res_df.append(rs[1])
        
          
    
    
        
    print("===================================")
    print("Time budeget: ",time_budget)
    [print(x) for x in results]
    
    res_df =  pd.DataFrame(res_df, columns = ["Dataset","Time","Preprocessing","AutoML","Accuracy","F1","Shape","PipeLine"])
    res_df.drop(columns=["PipeLine"]).to_csv("results/"+str(af_iters)+"_"+df_path[5:])
    res_df.to_csv("results/pipe_"+str(af_iters)+"_"+df_path[5:])
    
    return res_df




#df_path = "data/rcv1.csv" #crashes
#df_path = "data/20_newsgroups.csv"
#df_path = "data/gina.csv"
target_ft = "class"
#df_path = "data/airlines.csv"
#df_path = "data/dbworld-bodies.csv"
#df_path = "data/sonar.csv"
#df_path = "data/vehicle_sensIT.csv"

#df_path = "data/micro-mass.csv"
#df_path = "data/lymphoma_2classes.csv"
#df_path = "data/rsctc2010_3.csv"
#target_ft = "Decision"

#target_ft = "Delay"

#df_path = "data/GCM.csv"
#target_ft = "class"
#res = run_test(df_path, target_ft, mode=0 ,n_jobs=1,time_budget=3600)

#df_path = "data/AP_Omentum_Ovary.csv"
#target_ft = "Tissue"
#res = run_test(df_path, target_ft, mode=0 ,n_jobs=1,time_budget=3600)

#df_path = "data/dbworld-bodies-stemmed.csv"
#target_ft = "Class"

df_path = "data/20_newsgroups.csv"
for i in [5,10,20,30]:
    res = run_test(df_path, target_ft, mode=0 ,n_jobs=1,time_budget=3600 , af_iters=i)


# In[ ]:




