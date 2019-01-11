import pandas as pd
import fast_eda as eda
import numpy as np


def test_explore_general_german():
    file_path = 'E:/python/credit/input/'
    file_name = 'german-credit-scoring.csv'
    
    eda.setFileInfo(file_path,file_name)
    table = pd.read_csv(file_path+file_name,delimiter=';')
    table['TARGET'] = table['Score'].apply(lambda x: 1 if x=='bad' else 0)
    table['missing_1'] = table['Age in years'].apply(lambda x: 1 if x<40 else np.nan)
    table['missing_2'] = table['Housing'].apply(lambda x: 1 if x=='rent' else np.nan)
    
    
#    eda.explore_general(table, 'TARGET')
    table.drop(['Score'],axis=1,inplace=True)
    eda.explore_importance(table,'TARGET')
    eda.explore_glimpse(table)
    eda.explore_missing(table)
    table = eda.explore_preprocess(table)
    eda.explore_missing_col_corr(table,0.6)
    eda.explore_corr(table,'TARGET')
    eda.explore_dist(table,'TARGET')
#    table.drop(['Score'],axis=1,inplace=True)
#    eda.explore_importance(table,'TARGET')
    
test_explore_general_german()