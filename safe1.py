#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
import pandas as pd
import autosklearn.classification
import os
#import featuretools as ft
import warnings
import autofeat 
from tpot import TPOTClassifier
import json 
import time
from sklearn import preprocessing 
from datetime import datetime
import os

#warnings.filterwarnings('ignore')


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
        per_run_time_limit=time_budget//2,
        tmp_folder='./tmp/autosklearn_regression_example_tmp',
        output_folder='./tmp/autosklearn_regression_example_out',
        include_preprocessors=include_preprocessors,
        ml_memory_limit=240000,
        ensemble_memory_limit=240000,
        #metric=autosklearn.metrics.f1_weighted,
        #n_jobs=n_jobs
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
                                            #template="Selector-Transformer-Classifier",
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
    fsel = autofeat.AutoFeatClassifier(verbose=1,featsel_runs=af_iters,feateng_steps=1)
    X = fsel.fit_transform(X,y)
    return X
    
    
def run_test(df_path,target_ft, mode = 0, time_budget=30,n_jobs=-1, af_iters=5,nm_tara = "def"):
    now = datetime.now()

    current_time = now.strftime("%H:%M:%S")
    print("Start Time =", current_time)
    print("Time budget =", time_budget//60)
    print(df_path)
    results = []
    df = pd.read_csv(df_path)
    #object_columns = df.select_dtypes(include='object')
    #if len(object_columns.columns):
    #    df[object_columns.columns] = object_columns.apply(preprocessing.LabelEncoder().fit_transform)
    X = df.drop(columns=target_ft)
    y = df[target_ft]
    res_df = []
    autofeat_time = 0
    
    if mode ==0 or mode == 1 or mode == -1:
        start = time.monotonic()
        X_new = gen_feats_autofeat(X,y,af_iters=af_iters)
        print("X old shape:", X.shape)
        print("X new shape:", X_new.shape)
        end = time.monotonic()
        autofeat_time = int(end-start)
        start = time.monotonic()
        rs = run_as(X_new,y,target_ft, time_budget=time_budget, include_preprocessors=["no_preprocessing"], n_jobs=n_jobs)   
        #rs = run_tpot(X_new,y,target_ft, time_budget=time_budget, include_preprocessors=None, n_jobs=n_jobs)   
        end = time.monotonic()
        print("Actual Time Taken: ",str(end-start))
        results.append("AutoSK with Autofeat: " + rs[0])
        rs[1][0] = df_path[5:-4]
        rs[1][1] = str(round((time_budget+autofeat_time)/60,2))+'m'
        rs[1][2] = "AutoFeat"
        rs[1][3] = "AutoSK"
        rs[1][6] = str(X_new.shape)
        res_df.append(rs[1])

    
    if mode ==0 or mode == 2 or mode ==99:
        start = time.monotonic()
        #rs = run_tpot(X,y,target_ft, time_budget=time_budget+autofeat_time, include_preprocessors=False, n_jobs=n_jobs)   
        rs = run_as(X,y,target_ft, time_budget=time_budget+autofeat_time, include_preprocessors=["no_preprocessing"], n_jobs=n_jobs)   
        end = time.monotonic()
        print("Actual Time Taken: ",str(end-start))
        results.append("AutoSK Only with No Preprocessing: " + rs[0])
        rs[1][0] = df_path[5:-4]
        rs[1][1] = str(round((time_budget+autofeat_time)/60,2))+'m'
        rs[1][2] = "SAFE"
        rs[1][3] = "AutoSK"
        rs[1][6] = str(X.shape)
        res_df.append(rs[1])
        
        

    if mode ==0 or mode == 3:
        start = time.monotonic()
        #rs = run_tpot(X,y,target_ft, time_budget=time_budget+autofeat_time, include_preprocessors=True, n_jobs=n_jobs)   
        print(X.shape)
        print(y.shape)
        rs = run_as(X,y,target_ft, time_budget=time_budget+autofeat_time, include_preprocessors=None, n_jobs=n_jobs)   
        end = time.monotonic()
        print("Actual Time Taken: ",str(end-start))
        results.append("AutoSK Only with Preprocessing: " + rs[0])
        rs[1][0] = df_path[5:-4]
        rs[1][1] = str(round((time_budget+autofeat_time)/60,2))+'m'
        rs[1][2] = "AutoSK"
        rs[1][3] = "AutoSK"
        rs[1][6] = str(X.shape)
        res_df.append(rs[1])
        

    if mode ==0 or mode == 4 or mode == -1:
        #start = time.monotonic()
        #X_new = gen_feats_autofeat(X,y)
        #end = time.monotonic()
        #autofeat_time = int(end-start)
        #start = time.monotonic()
        rs = run_tpot(X_new,y,target_ft, time_budget=time_budget, include_preprocessors=None)   
        #end = time.monotonic()
        print("Actual Time Taken: ",str(end-start))
        results.append("TPOT with Autofeat: " + rs[0])
        rs[1][0] = df_path[5:-4]
        rs[1][1] = str(round((time_budget+autofeat_time)/60,2))+'m'
        rs[1][2] = "AutoFeat"
        rs[1][3] = "TPOT"
        rs[1][6] = str(X_new.shape)
        res_df.append(rs[1])
        
    
    
    if mode ==0 or mode == 5 or mode ==99:
        start = time.monotonic()
        rs = run_tpot(X,y,target_ft, time_budget=time_budget+autofeat_time, include_preprocessors=False)   
        end = time.monotonic()
        print("Actual Time Taken: ",str(end-start))
        results.append("TPOT Only with No Preprocessing: " + rs[0])
        rs[1][0] = df_path[5:-4]
        rs[1][1] = str(round((time_budget+autofeat_time)/60,2))+'m'
        rs[1][2] = "SAFE"
        rs[1][3] = "TPOT"
        rs[1][6] = str(X.shape)
        res_df.append(rs[1])
        
        

    if mode ==0 or mode == 6:
        start = time.monotonic()
        rs = run_tpot(X,y,target_ft, time_budget=time_budget+autofeat_time, include_preprocessors=True)   
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
    res_df.drop(columns=["PipeLine"]).to_csv("results/safe/"+str(af_iters)+"_both_at_"+nm_tara[5:])
    res_df.to_csv("results/safe/pipe_"+str(af_iters)+"_both_at_"+nm_tara[5:])
    
    return res_df




#df_path = "data/rcv1.csv" #crashes
#df_path = "data/20_newsgroups.csv"
#df_path = "data/gina.csv"
target_ft = "Class"
#df_path = "data/airlines.csv"
#df_path = "data/dbworld-bodies.csv"
df_path = "data/sonar.csv"
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
datas=[
        #("data/nomao.csv", "Class"),
        #("data/eeg_eye_state.csv", "Class"),
        ("data/Christine.csv", "class"),
        #("data/ailerons.csv", "binaryClass"),
        #("data/gina.csv", "class"),
        #("data/dbworld-bodies.csv","Class"),
        #("data/sonar.csv","Class"),
        #("data/lymphoma_2classes.csv","class"),
        #("data/rsctc2010_3.csv","Decision"),
        #("data/vehicle_sensIT.csv",'Y'),
        #("data/micro-mass.csv",'Class'),
        #("data/GCM.csv",'class'),
        #("data/AP_Omentum_Ovary.csv","Tissue"),
        #("data/dbworld-bodies-stemmed.csv","Class"),
        #("data/airlines.csv", "Delay"),
        
        ##("data/leukemia.csv","CLASS"),
        ##("data/AP_Endometrium_Prostate.csv","Tissue" ),
        ##("data/AP_Prostate_Lung.csv", "Tissue"),
        #("data/ovarianTumour.csv", "Decision"),
        #("data/arcene.csv", "Class"),
        #("data/yeast_ml8.csv", "class14"),
        #("data/madelon.csv", "Class"),
        #("data/eating.csv", "class"),
        #("data/hiva_agnostic.csv", "label"),
        #("data/anthracyclineTaxaneChemotherapy.csv", "Decision"),
        #("data/OVA_Breast.csv", "Tissue"),
        
        
        
        #("data/oh5.wc.csv", "CLASS_LABEL"),
        #("data/new3s.wc.csv", "CLASS_LABEL"),
        #("data/te23.wc.csv", "CLASS_LABEL"),
        
        ]


#datas=datas[1:2]
for each in datas:
    #try:
    df = pd.read_csv(each[0])
    df = df.rename(columns={each[1]:"Class"})
    object_columns = df.select_dtypes(include='object')
    if len(object_columns.columns):
        df[object_columns.columns] = object_columns.apply(preprocessing.LabelEncoder().fit_transform)
    df.to_csv("input.csv",index=False)
    try:
        os.remove("test.csv")
        pass
    except:
        pass
    start = time.monotonic()
    os.system("Rscript main.r") 
    end = time.monotonic()
    tm = end-start
    print(tm)
    #tm = 194
    #res = run_test(each[0], each[1], mode=-1 ,n_jobs=1,time_budget=600 , af_iters=5)
    res = run_test("test.csv", "class", mode=2,n_jobs=1,time_budget=600+int(tm) , af_iters=5,nm_tara = each[0])
    #except:
    #    pass


# In[ ]:




