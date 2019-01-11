import pandas as pd # package for high-performance, easy-to-use data structures and data analysis
import numpy as np # fundamental package for scientific computing with Python
import matplotlib
import matplotlib.pyplot as plt # for plotting
import seaborn as sns # for making plots with seaborn

import plotly.offline as py
import missingno as msno
import os
import gc

debug = False

def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df2 = df[categorical_columns]
    df = pd.get_dummies(df, dummy_na= nan_as_category)
    df = pd.concat([df2,df],axis=1)
    # df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


def explore_glimpse(dataframe):
    """
    Give first sight of the data, including data shape, column names, data 
    sample 
    
    Parameters
    ----------
    dataframe : pandas.Dataframe
        dataframe to explore.
    
    Output
    -------
    Statistic information is output as eda.txt
    Storage path is define with setFileInfo(file_path,file_name)
    """
    print('Size of dataframe data', dataframe.shape)
    print(dataframe.head(10))
    print(dataframe.columns.values)
    with open(sourcefilepath+textfilename, "a+") as text_file:
        print('Size of dataframe data', dataframe.shape, file=text_file)
        print(dataframe.head(10), file=text_file)
        print(dataframe.columns.values, file=text_file)

def explore_missing(dataframe, target=''):
    """
    Explore missing data with missingno, including matrix, heapmap and 
    some other interesting relationship.
    
    Parameters
    ----------
    dataframe : pandas.Dataframe
        dataframe with missing data.
    target : string, optional
        column name, target identifies some column which is used for 
        classification analyze.
        Relationship between missing status and target column is analyzed. 
    
    Output
    -------
    Images like matrix, heatmap.
    Statistic data is also produced.
    Storage path is define with setFileInfo(file_path,file_name)
    """
    
    total = dataframe.isnull().sum().sort_values(ascending = False)
    percent = (dataframe.isnull().sum()/dataframe.isnull().count()*100).sort_values(ascending = False)
    missing_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    print('missing rank',missing_data.head(40 if len(total)>40 else len(total)))

    if target != '':
        dataframe['incomplete'] = 1
        dataframe.loc[dataframe.isnull().sum(axis=1)/dataframe.isnull().count(axis=1)*100 < 35, 'incomplete'] = 0
        mean_c = np.mean(dataframe.loc[dataframe['incomplete'] == 0, target].values)
        mean_i = np.mean(dataframe.loc[dataframe['incomplete'] == 1, target].values)
        print('default ratio for more complete: {:.2} \ndefault ratio for less complete: {:.2}'.format(mean_c, mean_i))

    sample_size = min(dataframe.shape[0],500)
    msno.matrix(dataframe.sample(sample_size), inline=False, sparkline=True, figsize=(20,10), sort=None)
    plt.title('msno.matrix')
    plt.tight_layout
    plt.savefig(sourcefilepath+'msno.matrix.png')

    scale = dataframe.shape[1]/30+1
    fig_size = (20*scale,10*scale)    
    msno.heatmap(dataframe,fontsize=16,figsize=fig_size)
    plt.title('msno.heatmap')
    plt.tight_layout
    plt.savefig(sourcefilepath+'msno.heatmap.png')

    msno.dendrogram(dataframe, inline=False, fontsize=16, figsize=(40,20),orientation = 'top')
    plt.title('msno.dendrogram')
    plt.tight_layout
    plt.savefig(sourcefilepath+'msno.dendrogram.png')

    with open(sourcefilepath+textfilename, "a+") as text_file:
        print('missing rank',missing_data.head(40 if len(total)>40 else len(total)), file=text_file)
        if target != '':
            print('default ratio for more complete: {:.2} \ndefault ratio for less complete: {:.2}'.format(mean_c, mean_i), file=text_file)

def explore_preprocess(dataframe):
    """
    Process dataframe for better exploration. 
    Category columns are transformed with one hot encoder.
    Numeric columns with limited number of values are transformed as category.
    It's strong recommanded to use this method before any further exploration.
    Parameters
    ----------
    dataframe : pandas.Dataframe
        
    Return
    -------
    dataframe with additional columns after process
    """
    # dummy variable
    dataframe,new_cols = one_hot_encoder(dataframe, True)

    # value to category
    numeric_feats = [f for f in dataframe.columns if dataframe[f].dtype != object and f not in new_cols]
    for f_ in numeric_feats:
        if len(dataframe[f_].value_counts(dropna=False).index)<10:
            dataframe[f_+'_cat'] = dataframe[f_].astype(str)
    return dataframe

def explore_corr(dataframe,target=''):
    """
    Explore correlation amnong all number columns within table.
    As some columns are not number type, it's recommand to call explore_preprocess 
    before explore correlation.
    Correlation between target column and other columns is explicitly 
    analyzed if specified target as interesting column. 

    Parameters
    ----------
    dataframe : pandas.Dataframe
        dataframe to explore correlation
    target : string, optional
        Column name from dataframe for more exploration.
        
    Output
    -------
    Image of correlation heatmap 
    Correlation data with target column
    Storage path is define with setFileInfo(file_path,file_name)
    """
    corr = dataframe.corr()
    scale = dataframe.shape[1]/30+1
    fig_size = (20*scale,20*scale)
    plt.figure(figsize=fig_size)
    sns.heatmap(corr,linewidths=1,cmap='viridis_r')
    plt.title('pearson.correlation.heatmap')
    plt.tight_layout
    plt.savefig(sourcefilepath+'pearson.correlation.heatmap.png')
    if target != '':
        corr_target = corr.sort_values('TARGET', ascending = False)
        with open(sourcefilepath+textfilename, "a+") as text_file:
            # print out the correlation
            print('The Corr with target',corr_target[target].head(30), file=text_file)
 
def explore_missing_col_corr(dataframe, ratio=0.6,target=''):
    """
    Generate columns with missing status of each column and explore correlation
    between generated columns and original columns. With the additional columns
    ,more relationship become clear. Ex, missing satus of some column are 
    strong related with some column's value

    Parameters
    ----------
    dataframe : pandas.Dataframe
        dataframe to explore correlation
    ratio : float, optinal
        Value range between (0,1)
        Missing ratio threshold. If column missing ratio above this ratio
        , missing status column for that columns are generated 
    target : string, optional        
        Feature name from dataframe for more exploration.
        Correlation between target Feature and other Features is explicitly 
        analyzed. 
        
    Output
    -------
    Image of correlation heatmap, including missing status Features
    Text of correlation with target Feature
    Storage path is define with setFileInfo(file_path,file_name)
    """
    missing_total = dataframe.isnull().sum().sort_values(ascending = False)
    total = len(dataframe.iloc[:,0])
    missing_columns = [f_ for f_ in dataframe.columns.values if missing_total[f_]>(total*ratio)]
    missing_dataframe = dataframe[missing_columns].isnull().astype(float)
    # remain_columns = [f_ for f_ in dataframe.columns.values if f_ not in missing_columns]
    # remain_dataframe = dataframe[remain_columns]
    merged_dataframe = pd.concat([dataframe,missing_dataframe],axis=1)
    
    scale = dataframe.shape[1]/30+1
    fig_size = (20*scale,20*scale)  
    plt.figure(figsize=fig_size)
    corr = merged_dataframe.corr()
    # corr = remain_dataframe.corrwith(missing_dataframe)
    sns.heatmap(corr,linewidths=1,cmap='viridis_r')
    plt.title('pearson.correlation.heatmap.missing')
    plt.tight_layout
    plt.savefig(sourcefilepath+'pearson.correlation.heatmap.missing.png')

    if target != '':
        corr_target = corr.sort_values(target, ascending = False)
        # corr_target = corr_target[missing_columns]        
        with open(sourcefilepath+textfilename, "a+") as text_file:
            print('The missing.Corr with target',corr_target[target].head(30), file=text_file)

def explore_importance_features(dataframe, target, method='random forest'):
    from sklearn import preprocessing
    from sklearn.ensemble import RandomForestClassifier

    categorical_feats = [
        f for f in dataframe.columns if dataframe[f].dtype == 'object'
    ]

    for col in categorical_feats:
        lb = preprocessing.LabelEncoder()
        lb.fit(list(dataframe[col].values.astype('str')))
        dataframe[col] = lb.transform(list(dataframe[col].values.astype('str')))

    dataframe.fillna(-999, inplace = True)
    rf = RandomForestClassifier(n_estimators=50, max_depth=8, min_samples_leaf=4, max_features=0.5, random_state=2018)
    rf.fit(dataframe.drop([target],axis=1), dataframe[target])
    features = dataframe.drop([target],axis=1).columns.values

    importance_df = pd.DataFrame()
    importance_df['feature'] = features
    importance_df['importance'] = rf.feature_importances_
    importance_df.sort_values('importance',inplace=True,ascending=False)
    

def explore_importance(dataframe, target, method='random forest'):
    """
    Useful when you are checking a classification issue. 
    Provide importance list for predict target column.
    It support various classification algorithm, like 'random forest'
    Only binary target is supported
    Parameters
    ----------
    dataframe : pandas.Dataframe
        dataframe to explore importance

    target : string        
        Feature name, as predict result.
    method : string
        specified classification predict algorithm to rank feature importance.
        Tend to support logistic, random forest, xgboost,lgbm
    Output
    -------
    Image of importance rank
    Text of importance rank
    Storage path is define with setFileInfo(file_path,file_name)
    """

    from sklearn import preprocessing
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import RandomForestRegressor

    categorical_feats = [
        f for f in dataframe.columns if dataframe[f].dtype == 'object'
    ]

    for col in categorical_feats:
        lb = preprocessing.LabelEncoder()
        lb.fit(list(dataframe[col].values.astype('str')))
        dataframe[col] = lb.transform(list(dataframe[col].values.astype('str')))

    dataframe.fillna(-999, inplace = True)
    sample_size = min(1000, dataframe.shape[0])
    predict_classifier = dataframe[target].sample(sample_size).nunique() <= 2
        
    rf = RandomForestClassifier(n_estimators=50, max_depth=8, min_samples_leaf=4, max_features=0.5, random_state=2018)
    if predict_classifier == False:
        rf = RandomForestRegressor(n_estimators=50, max_depth=8, min_samples_leaf=4, max_features=0.5, random_state=2018)
    rf.fit(dataframe.drop([target],axis=1), dataframe[target])
    features = dataframe.drop([target],axis=1).columns.values

    importance_df = pd.DataFrame()
    importance_df['feature'] = features
    importance_df['importance'] = rf.feature_importances_
    importance_df.sort_values('importance',inplace=True,ascending=False)
    with open(sourcefilepath+textfilename, "a+") as text_file:
        print('The importance related with target',importance_df, file=text_file)
    
    plt.figure(figsize = (20, 40))
    plt.title("Feature Importance Rank")
    sns.barplot(y='feature',x='importance',data=importance_df)  
    plt.legend
    plt.savefig(sourcefilepath+'feature.importance.png')


fillna = '_fillna'
def kde_with_target_numeric(dataframe, feature, target):
    """
    Explore binary target based kernal distribution estimate.
    It explore different distribution for different target.
    Mean values for different kdes are also explored.
    Attention, only target with binary value is supported

    Parameters
    ----------
    dataframe : pandas.Dataframe
        dataframe to explore
    feature : string
        feature name, specified to explore which column's kde
    target : string        
        Feature name, only target with binary value is supported
        
    Output
    -------
    Image of target based kde
    Mean value for different kde are output
    Storage path is define with setFileInfo(file_path,file_name)
    """
    # Calculate the correlation coefficient between the new variable and the target
    corr = dataframe[target].corr(dataframe[feature])
    avg_0 = dataframe[dataframe[target] == 0][feature].dropna().median()
    avg_1 = dataframe[dataframe[target] == 1][feature].dropna().median()
    avg_0 = dataframe[dataframe[target] == 0][feature].dropna().mean()
    avg_1 = dataframe[dataframe[target] == 1][feature].dropna().mean()

    plt.figure(figsize = (12, 6))    
    # Plot the distribution for target == 0 and target == 1
    sns.kdeplot(dataframe.ix[dataframe[target] == 0, feature+fillna], label = 'TARGET == 0')
    sns.kdeplot(dataframe.ix[dataframe[target] == 1, feature+fillna], label = 'TARGET == 1')
    
    # label the plot
    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.title('kde %s Distribution' % feature)
    plt.legend()
    plt.tight_layout
    plt.savefig(sourcefilepath+feature.replace('/','_')+'.kde.png')
    
    with open(sourcefilepath+textfilename, "a+") as text_file:
        # print out the correlation
        print('The correlation between %s and the TARGET is %0.4f' % (feature, corr), file=text_file)
        # Print out average values
        print('Median value for  %s  that was 1 = %0.4f' % (feature, avg_1), file=text_file)
        print('Median value for  %s  that was 0 =     %0.4f' % (feature, avg_0), file=text_file)


def explore_numeric_dist(dataframe, feature, target=''):
    """
    Explore distribution of numeric type features, including whole feature 
    distribution, target based kde if target is specifiedd.
    Nan is explored as twice large value than the max value

    Parameters
    ----------
    dataframe : pandas.Dataframe
        dataframe to explore
    feature : string
        feature name which feature data is explored.
    target : string, optional        
        feature name, only feature with binary value is supported
    Output
    -------
    Image of feature data distribution
    Mean value for different kde are output
    Storage path is define with setFileInfo(file_path,file_name)
    """
    plt.figure(figsize=(12,6))
    plt.title("Distribution of "+ feature)
    new_feature = feature
    if dataframe[feature].isnull().sum() > 1:
        new_feature = dataframe.isnull().sum(axis=1)
        mean = dataframe[feature].dropna().mean()
        factor = -2 if mean>=0 else 2
        dataframe[new_feature] = dataframe[feature].fillna(factor*max(abs(dataframe[feature].dropna())))

    ax = sns.distplot(dataframe[new_feature])
    plt.tight_layout
    plt.savefig(sourcefilepath+feature.replace('/','_')+'.dist.png')

    if target != '' and feature != target and dataframe[target].nunique == 2:
        kde_with_target_numeric(dataframe, feature, target)


def dist_with_target_nominal(dataframe, feature, target):
    """
    Explore binary target based nominal type feature data distribution.
    It try to find different distribution for different target.
    Additionally, Chi square between feature and target are explored.

    Parameters
    ----------
    dataframe : pandas.Dataframe
        dataframe to explore.
    feature : string
        feature name, specified to explore which column's distribution
    target : string        
        feature name, which is often target column
        
    Output
    -------
    Image of target based distribution
    Chi square p-value is output
    Storage path is define with setFileInfo(file_path,file_name)
    """
    from scipy.stats import chi2_contingency
    props = pd.crosstab(dataframe[feature], dataframe[target],dropna=False)
    c = chi2_contingency(props, lambda_="log-likelihood")
    print(props, "\n p-value= ", c[1])
    with open(sourcefilepath+textfilename, "a+") as text_file:
        # print out the correlation
        print('The chi square props between %s and the TARGET is %0.4f' % (feature, c[1]), file=text_file)
    
    plt.figure(figsize = (20, 10))
    plt.title("Distribution of "+ feature)
    ax = sns.countplot(x=feature, hue=target, data=dataframe[[feature,target]].fillna('NaN_Fill'))
    plt.legend
    # plt.plot()
    plt.tight_layout
    plt.savefig(sourcefilepath+feature.replace('/','_')+'.histogram.png')

def explore_nominal_dist(dataframe, feature, target=''):
    """
    Explore distribution of nominal type features, including whole feature 
    distribution, target based distribution if target is specifiedd.
    Parameters
    ----------
    dataframe : pandas.Dataframe
        dataframe to explore
    feature : string
        feature name, specified to explore which column's distribution
    target : string, optional        
        Feature name, based on this,different distribution are explored
        
    Output
    -------
    Image of feature data distribution
    Chi square p-value is output
    Storage path is define with setFileInfo(file_path,file_name)
    """
    temp = dataframe[feature].value_counts(dropna=False)
    plt.rcParams.update({'font.size': 22})
    plt.figure(figsize=(10,10))
    plt.title("Distribution of "+ feature)
    plt.pie(temp.values,labels=temp.index,autopct='%1.1f%%',shadow=True)
    #plt.tight_layout
    plt.savefig(sourcefilepath+feature.replace('/','_')+'.pie.png')

    if target != '' and feature != target and dataframe[target].nunique == 2:
        dist_with_target_nominal(dataframe, feature, target)   

def explore_dist(dataframe,target=''):
    """
    Explore data distribution of features from dataframe.
    Parameters
    ----------
    dataframe : pandas.Dataframe
        dataframe to explore
    target : string, optional        
        Feature name, based on this,different distribution are explored
        
    Output
    -------
    Image of feature data distribution
    Chi square p-value or mean of kde is output
    Storage path is define with setFileInfo(file_path,file_name)
    """
    for f_ in dataframe.columns.values:
        if dataframe[f_].dtype == 'object':
            explore_nominal_dist(dataframe, f_, target)
        else:
            explore_numeric_dist(dataframe, f_, target)

def explore_aggr(dataframe, base_col, aggr_cols='', interval=1):
    """
    Explore data with series baseline, like time series.
    Aggregation as 'min', 'max', 'size','mean','sum' are supported.
    Parameters
    ----------
    dataframe : pandas.Dataframe
        dataframe to explore
    base_col : string
        feature name, as series baseline
    aggr_cols : string array, optional
        Feature name array, which are explored with base_col based aggregation.
        Only number features are supported.
        If not specified, all features except base_col are explored.
    interval : int, optional
        Use to control series desity. Cluster neighour data give clear view.
    Output
    -------
    Image of feature data distribution
    Storage path is define with setFileInfo(file_path,file_name)
    """
    if aggr_cols == '':
        aggr_cols = [col for col in dataframe.columns]
    else:
        aggr_cols.append(base_col)
        
    
    dataframe_ = dataframe
    if interval>1:
        dataframe_[base_col] = round(dataframe_[base_col]/interval)
    # if aggr_cols.count > 10:
    #     return
    dataframe_enc = dataframe_[aggr_cols]
    dataframe_enc, col_cat = one_hot_encoder(dataframe_enc, True)

    aggregations = {}
    #{'MONTHS_BALANCE': ['min', 'max', 'size','mean','sum']}
    numeric_cols = [col for col in dataframe_enc.columns if dataframe_enc[col].dtype != 'object' and col != base_col]
    # aggregations[numeric_cols] = ['min', 'max', 'size','mean','sum']
    for col in numeric_cols:
        aggregations[col] = ['min', 'max', 'size','mean','sum']
    dataframe_agg = dataframe_enc.groupby(base_col).agg(aggregations)
    dataframe_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in dataframe_agg.columns.tolist()])
    dataframe_agg.reset_index(inplace=True)
    dataframe_agg.sort_values(by = base_col,inplace=True)

    for col in dataframe_agg.columns:
        plt.figure(figsize=(12,6))
        plt.title("Time series of "+ base_col+'_'+col)
        plt.title("Time series of "+ base_col)
        sns.lineplot(x=base_col, y=col, data=dataframe_agg)
        plt.savefig(sourcefilepath+base_col.replace('/','_')+'.'+col.replace('/','_')+'.line.png')

    del dataframe_enc, dataframe_agg
    gc.collect()

def explore_scatter(dataframe, axis_col, scatter_cols=''):
    """
    Explore data along axis_col, like time or some order.
    
    Parameters
    ----------
    dataframe : pandas.Dataframe
        dataframe to explore
    axis_col : string
        feature name, as series axis
    scatter_cols : string array, optional
        Feature name array, which are explored with axis_col.
        Only number features are supported.
        If not specified, all features except base_col are explored.

    Output
    -------
    Image of feature data scatter
    Storage path is define with setFileInfo(file_path,file_name)
    """
    if scatter_cols == '':
        scatter_cols = [col for col in dataframe.columns]
    else:
        scatter_cols.append(axis_col)
        
    
    # if aggr_cols.count > 10:
    #     return
    df = dataframe[scatter_cols]

    for col in df.columns:
        if col == axis_col or df[col].dtype=='object':
            continue        
        plt.figure(figsize=(12,6))
        plt.title("Sequence of "+ axis_col+'_'+col)
        plt.xlabel(axis_col, size = 22);
        plt.ylabel(col, size = 22)
        min_x = np.min(df[axis_col])
        max_x = np.max(df[axis_col])
        plt.xlim((min_x, max_x))
        plt.xticks(np.arange(min_x, max_x, (max_x-min_x)/10))
        plt.plot(df[axis_col], df[col],  'bo', alpha = 0.5)
        y_line = df[col][np.argmin(df[axis_col])]
        vertical_len = df[axis_col].nunique()
        plt.hlines(y_line, 0, vertical_len, linestyles = '--', colors = 'r')
        plt.savefig(sourcefilepath+axis_col.replace('/','_')+'.'+col.replace('/','_')+'.scatter.png')

    gc.collect()


sourcefilename='somefile.csv'
sourcefilepath='e://eda/'
textfilename='eda.txt' 

def setFileInfo(filepath, filename):
    """
    Configure data file path and file name. 
    This must be used before other method, as it generate exploration result 
    storage.
    Parameters
    ----------
    filepath : string
        data file path
    filename : string
        data file path
    Output
    -------
    Generate path filepath/eda/filename/ to storage result.    
    """
    global sourcefilename
    global sourcefilepath

    sourcefilename = filename[:-4]
    sourcefilepath = filepath + 'eda/'
    if not os.path.exists(sourcefilepath):
        os.mkdir(sourcefilepath) 
    sourcefilepath = sourcefilepath + sourcefilename + '/'
    if not os.path.exists(sourcefilepath):
        os.mkdir(sourcefilepath) 


def explore_general(dataframe, target=''):
    """
    Explore table data generally with specified dataframe, output result
    including missing, correlation, number / nominal data distribution.
    If you are exploring for a binary classification problem, specified target
    name will give a closer watch on the target, like correlation with other 
    columns and importance weight of each column.

    Parameters
    ----------
    dataframe : pandas.Dataframe
        dataframe to explore
    target : string, optional
        target identifies focused column.
        Correlation between target column and other columns is explicitly 
        analyzed. 
        
    Output
    -------
    Image of  missing, correlation, number / nominal data distribution, stored
    under filepath/EDA/filename.
    Statistic info is stored in filepath/EDA/filename/eda.txt
    """
    explore_glimpse(dataframe)
    total = dataframe.isnull().sum().sort_values(ascending = False)
    if total[0] > 0:
        explore_missing(dataframe, target)
    dataframe = explore_preprocess(dataframe)
    explore_missing_col_corr(dataframe,0.6,target)
    explore_corr(dataframe)
#    if target != '':
#        df = dataframe.copy(deep=True)
#        explore_importance(df,target)
    explore_importance(dataframe,target)
    explore_dist(dataframe, target)