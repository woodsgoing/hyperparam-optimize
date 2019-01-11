# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 09:12:12 2018

@author: june
"""

import pandas as pd
import numpy as np
import os
import time
import dask.dataframe as dd
import csv
from sklearn.model_selection import KFold
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

_DEBUG = True
global outputfilename
outputfilename=''
global outputfilepath
outputfilepath=''
global out_trial_file_name
out_trial_file_name=''
global time_line
time_line='.'

CLASS_NUM_BIN = 2

def setEnvInfo(filepath, filename):
    """
    Configure data file path and file name. 
    This must be used before other method, as it generate log info storage path.
    Parameters
    ----------
    filepath : string
        log file path
    filename : string
        log file name
    Output
    -------
    Generate path filepath/filename/ to storage result.    
    """
    global outputfilename
    global outputfilepath
    global out_trial_file_name
    outputfilename = filename
    outputfilepath = filepath
    out_trial_file_name = outputfilepath+'xxx_trials.csv'
    if not os.path.exists(outputfilepath):
        os.mkdir(outputfilepath) 
    global time_line
    time_line = time.strftime("%Y_%m_%d", time.localtime()) 

def _log(*arg, mode):
    global outputfilename
    global outputfilepath
    if outputfilename == '' or outputfilepath == '':
        return  
    with open(outputfilepath+outputfilename+mode+time_line+'.bayes.opt', "a+") as text_file:
        print(*arg, file=text_file)

def trace(*arg):
    _log(*arg, mode='trace')

def debug(*arg):
    if _DEBUG == True:
        _log(*arg, mode = 'debug')


# Optimize
def bayes_optimize(dataframe, target, model, space, max_evals = 1000, eda=True):
    """
    Search for optimized parameters based on bayes TPE (Tree-struct Parzen 
    estimator.
    
    Parameters
    ----------
    dataframe : pandas.Dataframe
        dataframe to process.
    target : string
        Feature name, target specifies some feature which is used for 
        prediction analyze.
    model : function
        There are some predefined model functions inside fast_model. Self-defined 
        functions can also be used with same interface and return.
    space : function
        There are some parameter space functions predefined in fast_model, 
        pair with each model. Self-defined parameter space functions are 
        acceptable.
    max_evals : int, optional
        Integer value large than 1. It specifies maximum parameter search times.
    eda : bool, optional
        Plot charactors of parameters which is used in bayes estimation when True. 
    
    Output
    -------
    Parameter set together with scores. With best score, optimized parameters 
    can be selected.
    Parameter charactors imagess as distribution and scattor plot.
    
    """
    global train_set
    train_set = dataframe      
    global train_model
    train_model = model
    global train_target
    train_target = target   
    global out_trial_file_name
    global time_line
    out_trial_file_name = model.__name__+'.'+target+'.'+time_line+'.trials.csv'

    is_classifier = dataframe[target].sample(1000).nunique()<=CLASS_NUM_BIN\
    or dataframe[target].dtype=='object'
    train_space = space(is_classifier)
    bayes_trials = Trials()  
    
    of_connection = open(outputfilepath+out_trial_file_name, 'w')
    writer = csv.writer(of_connection)
    writer.writerow(['loss', 'params', 'iteration', 'train_time'])
    of_connection.close()    

    best = fmin(fn = _objective, \
                space = train_space, \
                algo = tpe.suggest, 
                max_evals = max_evals, \
                trials = bayes_trials)

    bayes_trials_results = sorted(bayes_trials.results, key = lambda x: x['loss'])
    trace(bayes_trials_results)
    if eda == True:
        _hyper_opt_eda()



#import lightgbm as lgbm
from timeit import default_timer as timer
global ITERATION
ITERATION = 0

global train_set
train_set=pd.DataFrame()

def _objective(params):
    """Objective function for Gradient Boosting Machine Hyperparameter Optimization"""
    global train_set
    if train_set.shape[0]==0:
        return

    global train_target
    global train_model
    global ITERATION    
    ITERATION += 1

    start = timer()
    score = train_model(params, dataframe=train_set, target=train_target)
    run_time = timer() - start    

    loss = 1 - score
    # Write to the csv file ('a' means append)
    of_connection = open(outputfilepath+out_trial_file_name, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([loss, params, ITERATION, run_time])

    return {'loss': loss, 'params': params, 'iteration': ITERATION,
#            'estimators': n_estimators, 
            'train_time': run_time, 'status': STATUS_OK}

def _hyper_opt_eda():
    import fast_eda

    table = pd.read_csv(outputfilepath+out_trial_file_name)
    from ast import literal_eval
    dic = literal_eval(table['params'][0])
    for key in dic.keys():
        table[key] =  table['params'].apply(lambda x: literal_eval(x)[key])
    
    table.drop(['params','train_time'], axis=1, inplace=True)
    global train_model
    global train_target    
    fast_eda.setFileInfo(outputfilepath, train_model.__name__+train_target+outputfilename)
    fast_eda.explore_general(table,target='loss')
    fast_eda.explore_scatter(table,axis_col='iteration')
    fast_eda.explore_scatter(table,axis_col='loss')
 