import pandas as pd
import fast_hyper_opt as hyopt
import matplotlib.pyplot as plt # for plotting
import seaborn as sns 
import gc
import time
import fast_model
from contextlib import contextmanager
@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

in_file_path = 'E:/python/credit/input/'
out_file_path = 'E:/python/credit/output/'
LOOP_NUM = 1
LOOP_NUM_LIMITED = 2
EDA_FLAG = False

def test_hyper_opt_general():
    hyopt.setEnvInfo(out_file_path,'application_train.log')
    fast_model.setEnvInfo(out_file_path,'application_train.log')
    table = pd.read_csv(in_file_path+'application_train_sample.csv')
#    table = pd.read_csv(in_file_path+'application_train.csv')
    table.reset_index(drop=True,inplace=True)
    test_lgbm_bayes_opt(table)
    test_random_forest_bayes_opt(table)
    test_KNN_bayes_opt(table)
    test_linear_regression_opt(table)

def test_lgbm_bayes_opt(table):
    with timer("hyper opt classify lgbm application_train"):
        df = table.copy(deep=True)
#        hyopt.bayes_optimize(df, target='NAME_INCOME_TYPE', max_evals=LOOP_NUM, model=fast_model.int_module_lgbm_cv, space=fast_model.int_module_lgbm_space, eda=EDA_FLAG)
        hyopt.bayes_optimize(df, target='TARGET', max_evals=LOOP_NUM, model=fast_model.int_module_lgbm_cv, space=fast_model.int_module_lgbm_space, eda=EDA_FLAG)
#        df.to_csv(out_file_path+'application_train_sample.lgbm.hyperopt1226.output.csv')
        del df
        gc.collect()

    with timer("hyper opt regression lgbm application_train"):
        df = table.copy(deep=True).drop(['TARGET'],axis=1)
        df = df[df['AMT_CREDIT'].notnull()]
        df.reset_index(drop=True,inplace=True)
        hyopt.bayes_optimize(df, target='AMT_CREDIT', max_evals=LOOP_NUM, model=fast_model.int_module_lgbm_cv, space=fast_model.int_module_lgbm_space, eda=EDA_FLAG)
#        df.to_csv(out_file_path+'application_train_sample.lgbm.hyperopt1226.output.csv')
        del df
        gc.collect()


    with timer("hyper opt multi-classify lgbm application_train"):
        df = table.copy(deep=True).drop(['TARGET'],axis=1)
        df = df[df['OCCUPATION_TYPE'].notnull()]
        df.reset_index(drop=True,inplace=True)
        hyopt.bayes_optimize(df, target='OCCUPATION_TYPE', max_evals=LOOP_NUM, model=fast_model.int_module_lgbm_cv, space=fast_model.int_module_lgbm_space, eda=EDA_FLAG)
#        df.to_csv(out_file_path+'application_train_sample.lgbm.hyperopt1226.output.csv')
        del df
        gc.collect()

    with timer("hyper opt multi-integrate lgbm application_train"):
        df = table.copy(deep=True).drop(['TARGET'],axis=1)
        df = df[df['AMT_REQ_CREDIT_BUREAU_WEEK'].notnull()]
        df.reset_index(drop=True,inplace=True)
        hyopt.bayes_optimize(df, target='AMT_REQ_CREDIT_BUREAU_WEEK', max_evals=LOOP_NUM, model=fast_model.int_module_lgbm_cv, space=fast_model.int_module_lgbm_space, eda=EDA_FLAG)
#        df.to_csv(out_file_path+'application_train_sample.lgbm.hyperopt1226.output.csv')
        del df
        gc.collect()

def test_random_forest_bayes_opt(table):
    with timer("hyper opt classify random_forest application_train"):
        df = table.copy(deep=True)
        hyopt.bayes_optimize(df, target='TARGET', max_evals=LOOP_NUM, model=fast_model.int_module_random_forest, space=fast_model.int_module_random_forest_space, eda=EDA_FLAG)
        del df
        gc.collect()
    with timer("hyper opt regression random_forest application_train"):
        df = table.copy(deep=True).drop(['TARGET'],axis=1)
        hyopt.bayes_optimize(df, target='AMT_CREDIT', max_evals=LOOP_NUM, model=fast_model.int_module_random_forest, space=fast_model.int_module_random_forest_space, eda=EDA_FLAG)
        del df
        gc.collect()
    with timer("hyper opt multi-classify random_forest application_train"):
        df = table.copy(deep=True).drop(['TARGET'],axis=1)
        df = df[df['OCCUPATION_TYPE'].notnull()]
        hyopt.bayes_optimize(df, target='OCCUPATION_TYPE', max_evals=LOOP_NUM, model=fast_model.int_module_random_forest, space=fast_model.int_module_random_forest_space, eda=EDA_FLAG)
        del df
        gc.collect()
    with timer("hyper opt multi-integrate random_forest application_train"):
        df = table.copy(deep=True).drop(['TARGET'],axis=1)
        df = df[df['AMT_REQ_CREDIT_BUREAU_WEEK'].notnull()]
        df.reset_index(drop=True,inplace=True)
        hyopt.bayes_optimize(df, target='AMT_REQ_CREDIT_BUREAU_WEEK', max_evals=LOOP_NUM, model=fast_model.int_module_random_forest, space=fast_model.int_module_random_forest_space, eda=EDA_FLAG)
#        df.to_csv(out_file_path+'application_train_sample.lgbm.hyperopt1226.output.csv')
        del df
        gc.collect()

def test_KNN_bayes_opt(table):
    with timer("hyper opt classify KNN regression application_train"):
        df = table.copy(deep=True)
        hyopt.bayes_optimize(df, target='TARGET', max_evals=LOOP_NUM, model=fast_model.int_module_knn, space=fast_model.int_module_knn_space, eda=EDA_FLAG)
        del df
        gc.collect()

    with timer("hyper opt regression KNN application_train"):
        df = table.copy(deep=True).drop(['TARGET'],axis=1)
        hyopt.bayes_optimize(df, target='AMT_CREDIT', max_evals=LOOP_NUM, model=fast_model.int_module_knn, space=fast_model.int_module_knn_space, eda=EDA_FLAG)
        del df
        gc.collect()

    with timer("hyper opt multi-classify KNN regression application_train"):
        df = table.copy(deep=True).drop(['TARGET'],axis=1)
        df = df[df['OCCUPATION_TYPE'].notnull()]
        hyopt.bayes_optimize(df, target='OCCUPATION_TYPE', max_evals=LOOP_NUM, model=fast_model.int_module_knn, space=fast_model.int_module_knn_space, eda=EDA_FLAG)
        del df
        gc.collect()
    with timer("hyper opt multi-integrate KNN application_train"):
        df = table.copy(deep=True).drop(['TARGET'],axis=1)
        df = df[df['AMT_REQ_CREDIT_BUREAU_WEEK'].notnull()]
        hyopt.bayes_optimize(df, target='AMT_REQ_CREDIT_BUREAU_WEEK', max_evals=LOOP_NUM, model=fast_model.int_module_knn, space=fast_model.int_module_knn_space, eda=EDA_FLAG)
#        df.to_csv(out_file_path+'application_train_sample.lgbm.hyperopt1226.output.csv')
        del df
        gc.collect()
        
def test_linear_regression_opt(table):
    with timer("hyper opt classify logistic application_train"):
        df = table.copy(deep=True)
        hyopt.bayes_optimize(df, target='TARGET', max_evals=LOOP_NUM, model=fast_model.int_module_linear_regression, space=fast_model.int_module_linear_regression_space, eda=EDA_FLAG)
        del df
        gc.collect()

    with timer("hyper opt regression linear application_train"):
        df = table.copy(deep=True).drop(['TARGET'],axis=1)
        df = df[df['AMT_CREDIT'].notnull()]
        hyopt.bayes_optimize(df, target='AMT_CREDIT', max_evals=LOOP_NUM_LIMITED, model=fast_model.int_module_linear_regression, space=fast_model.int_module_linear_regression_space, eda=EDA_FLAG)
        del df
        gc.collect()
        
    with timer("hyper opt multi-classify logistic application_train"):
        df = table.copy(deep=True).drop(['TARGET'],axis=1)
        df = df[df['OCCUPATION_TYPE'].notnull()]
        hyopt.bayes_optimize(df, target='OCCUPATION_TYPE', max_evals=LOOP_NUM, model=fast_model.int_module_linear_regression, space=fast_model.int_module_linear_regression_space, eda=EDA_FLAG)
        del df
        gc.collect()
    with timer("hyper opt multi-integrate linear regression application_train"):
        df = table.copy(deep=True).drop(['TARGET'],axis=1)
        df = df[df['AMT_REQ_CREDIT_BUREAU_WEEK'].notnull()]
        hyopt.bayes_optimize(df, target='AMT_REQ_CREDIT_BUREAU_WEEK', max_evals=LOOP_NUM, model=fast_model.int_module_linear_regression, space=fast_model.int_module_linear_regression_space, eda=EDA_FLAG)
        del df
        gc.collect()

        
def test_bayes_opt_eda():
    hyopt.setEnvInfo(out_file_path,'application_train.log')
    hyopt._hyper_opt_eda()

test_hyper_opt_general()

#test_bayes_opt_eda()