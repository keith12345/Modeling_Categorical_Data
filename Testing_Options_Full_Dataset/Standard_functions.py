from collections import OrderedDict

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB

 #####################
 # Plotting Features #
 #####################

def plot_features(df, alpha=.3, sample_size=500):
    """
    Takes a Dataframe as an input.
    Drops columns unnecessary for visualizations:
        product_id
        user_id
        latest_cart
    Plots engineered features.
    """
    
    sample = (df.drop(['product_id','user_id','latest_cart'],axis=1)
                .sample(1000, random_state=44)) 
    plt.figure(figsize=(9,9))
    sns.pairplot(sample,hue='in_cart', plot_kws=dict(alpha=alpha, edgecolor='none'))


 #####################
 # Train, Val, Split #
 #####################

 # Due to the nature of our data we must be careful 
 # with how we split it and have therefore created 
 # our own function for doing so.
    
def get_user_split_data(df, val_size=.2, seed=42):
    """
    We will create an 80/20 split of users and take all orders for those users.  
    Default Values:
        Test Size:
            80 - Training Data
            20 - Validation Data
        Random Seed:
            42
    From that we will take the in_cart column created in the 'Preparing our
    Test Data' section which will be used as our target variable.
    
    test_size - takes any value between 0 and 1
    seed - takes any integer.
    Outputs:
    X_tr, X_val, y_tr, y_val
    """

    rs = np.random.RandomState(seed)
    
    total_users = df['user_id'].unique() 
    # Multiplies the number of observations (user_id's) by the test size
    # to get a list of validation users.
    val_users = rs.choice(total_users, 
                   size=int(total_users.shape[0] * val_size), 
                   replace=False)

    df_tr = df[~df['user_id'].isin(val_users)]
    df_val = df[df['user_id'].isin(val_users)] 

    y_tr, y_val = df_tr['in_cart'], df_val['in_cart']
    X_tr = df_tr.drop(['product_id','user_id','latest_cart','in_cart'],axis=1) 
    X_val = df_val.drop(['product_id','user_id','latest_cart','in_cart'],axis=1)

    return X_tr, X_val, y_tr, y_val


 ############
 # Show All #
 ############
    
 # Quick shortcut so that we don't need to enter 
 # a bunch of functions every time we add a new
 # feature and want to see how it performs.

def plot_fit_score_pred(df, X_tr, X_val, y_tr, y_val):
    """    
    Takes a DataFrame, training, and validation data as its input.
    Returns Seaborn Pairplot, f1-score, features and their coefficients, and predicted non-re-orders and re-orders.
    """
    
    # Note that plot_features already removes 'product_id','user_id',
    # and 'latest_cart' so we don't need to do it for that function.
    
    plot_features(df)
    plt.show()
    
    reduced_df = df.drop(['product_id','user_id',
                        'latest_cart','in_cart'],axis=1)
    

    
    features = reduced_df.columns
    
    lr = LogisticRegression(solver='lbfgs')
    lr.fit(X_tr, y_tr)
    vals = pd.DataFrame(lr.predict(X_val))[0].value_counts()
    coefs = [round(x,4) for x in lr.coef_.tolist()[0]]
    print('Our f1-score is',f1_score(lr.predict(X_val), y_val))
    print('The coefficients are: \n',
          pd.DataFrame(list(zip(features,coefs)),
                columns=['Features','Coefficients']))
    print('And we\'ve predicted',vals[0],'non-re-orders and',
    vals[1],'re-orders.')
    
def fit_score_pred_log(df, X_tr, X_val, y_tr, y_val):
    """    
    Takes a DataFrame, training, and validation data as its input.
    Returns f1-score, features and their coefficients, and predicted non-re-orders and re-orders.
    """
    
    reduced_df = df.drop(['product_id','user_id',
                        'latest_cart','in_cart'],axis=1)
    
    features = reduced_df.columns
    
    lr = LogisticRegression(solver='lbfgs')
    lr.fit(X_tr, y_tr)
    vals = pd.DataFrame(lr.predict(X_val))[0].value_counts()
    coefs = [round(x,4) for x in lr.coef_.tolist()[0]]
    print('Our f1-score is',f1_score(lr.predict(X_val), y_val))
    print('The coefficients are: \n',
          pd.DataFrame(list(zip(features,coefs)),
                columns=['Features','Coefficients']))
    print('And we\'ve predicted',vals[0],'non-re-orders and',
    vals[1],'re-orders.')
    

def kfold_val_fit_score_pred_log(df, val_size=.2, seed=42):
    
    df = df.drop(['product_id','latest_cart'],axis=1)
    
    ids = pd.DataFrame(df.user_id)
    
    kf = KFold(n_splits=5, shuffle=True, random_state = seed)
    model_results = [] #collect the validation results for both models
    
    for train_ids, val_ids in kf.split(ids,ids):
        
        X_train, y_train = df.iloc[train_ids], df.iloc[train_ids]
        X_val, y_val = df.iloc[val_ids], df.iloc[val_ids] 
        
        X_train = pd.DataFrame(X_train).drop(['in_cart','user_id'],axis=1)
        y_train = pd.DataFrame(y_train).in_cart
        X_val = pd.DataFrame(X_val).drop(['in_cart','user_id'],axis=1)
        y_val = pd.DataFrame(y_val).in_cart
        
        lr = LogisticRegression(solver='liblinear')
        lr.fit(X_train, y_train)
        vals = pd.DataFrame(lr.predict(X_val))[0].value_counts()
        coefs = [round(x,4) for x in lr.coef_.tolist()[0]]
    
        model_results.append(f1_score(lr.predict(X_val), y_val))
        
    print('Individual f-1 score: ', model_results)
    print(f'Average f1-score: {np.mean(model_results):.3f} +- {np.std(model_results):.3f}')
    
    
def kfold_val_fit_score_pred_G_NB(df, val_size=.2, seed=42):
    
    df = df.drop(['product_id','latest_cart'],axis=1)
    
    ids = pd.DataFrame(df.user_id)
    
    kf = KFold(n_splits=5, shuffle=True, random_state = seed)
    model_results = [] #collect the validation results for both models
        
    for train_ids, val_ids in kf.split(ids,ids):
        print(1)
        X_train, y_train = df.iloc[train_ids], df.iloc[train_ids]
        X_val, y_val = df.iloc[val_ids], df.iloc[val_ids] 
        
        X_train = pd.DataFrame(X_train).drop(['in_cart','user_id'],axis=1)
        y_train = pd.DataFrame(y_train).in_cart
        X_val = pd.DataFrame(X_val).drop(['in_cart','user_id'],axis=1)
        y_val = pd.DataFrame(y_val).in_cart
        
        print(X_train.columns.tolist())
        
        clf = GaussianNB(var_smoothing=1e-9)
        clf.fit(X_train, y_train)
        vals = pd.DataFrame(clf.predict(X_val))[0].value_counts()
    
        model_results.append(f1_score(clf.predict(X_val), y_val))
        
    print('Individual f-1 score: ', model_results)
    print(f'Average f1-score: {np.mean(model_results):.3f} +- {np.std(model_results):.3f}') 
    
    
def kfold_val_fit_score_pred_M_NB(df, val_size=.2, seed=42):
    
    df = df.drop(['product_id','latest_cart'],axis=1)
    
    df.diff_between_average_and_current_order_time = (
        df.diff_between_average_and_current_order_time 
        + abs(df.diff_between_average_and_current_order_time.min()))
    
    ids = pd.DataFrame(df.user_id)
    
    kf = KFold(n_splits=5, shuffle=True, random_state = seed)
    model_results = [] #collect the validation results for both models
        
    for train_ids, val_ids in kf.split(ids,ids):
        print(1)
        X_train, y_train = df.iloc[train_ids], df.iloc[train_ids]
        X_val, y_val = df.iloc[val_ids], df.iloc[val_ids] 
        
        X_train = pd.DataFrame(X_train).drop(['in_cart','user_id'],axis=1)
        y_train = pd.DataFrame(y_train).in_cart
        X_val = pd.DataFrame(X_val).drop(['in_cart','user_id'],axis=1)
        y_val = pd.DataFrame(y_val).in_cart
        
        clf = MultinomialNB()
        clf.fit(X_train, y_train)
        vals = pd.DataFrame(clf.predict(X_val))[0].value_counts()
    
        model_results.append(f1_score(clf.predict(X_val), y_val))
        
    print('Individual f-1 score: ', model_results)
    print(f'Average f1-score: {np.mean(model_results):.3f} +- {np.std(model_results):.3f}')     
    
    
    
def kfold_val_fit_score_pred_RF(df, val_size=.2, seed=42):
    
    df = df.drop(['product_id','latest_cart'],axis=1)
    
    ids = pd.DataFrame(df.user_id)
    
    kf = KFold(n_splits=5, shuffle=True, random_state = seed)
    model_results = [] #collect the validation results for both models
        
    for train_ids, val_ids in kf.split(ids,ids):
        print(1)
        X_train, y_train = df.iloc[train_ids], df.iloc[train_ids]
        X_val, y_val = df.iloc[val_ids], df.iloc[val_ids] 
        
        X_train = pd.DataFrame(X_train).drop(['in_cart','user_id'],axis=1)
        y_train = pd.DataFrame(y_train).in_cart
        X_val = pd.DataFrame(X_val).drop(['in_cart','user_id'],axis=1)
        y_val = pd.DataFrame(y_val).in_cart
        
        rfc = RandomForestClassifier(n_estimators=50)
        rfc.fit(X_train, y_train)        
        vals = pd.DataFrame(rfc.predict(X_val))[0].value_counts()
        model_results.append(f1_score(rfc.predict(X_val), y_val))
        
    print('Individual f-1 score: ', model_results)
    print(f'Average f1-score: {np.mean(model_results):.3f} +- {np.std(model_results):.3f}') 
    

    
    
def fit_score_pred_RF(df, X_tr, X_val, y_tr, y_val):
    """    
    Takes a DataFrame, training, and validation data as its input.
    Returns f1-score, features and their coefficients, and predicted non-re-orders and re-orders.
    """
    
    rfc = RandomForestClassifier(n_estimators=10)
    rfc.fit(X_train, y_train)        
    vals = pd.DataFrame(rfc.predict(X_val))[0].value_counts()
    print('Our f1-score is',f1_score(rfc.predict(X_val), y_val))
    print('And we\'ve predicted',vals[0],'non-re-orders and',
    vals[1],'re-orders.')
    
def fit_score_pred_G_NB(X_tr, X_val, y_tr, y_val):
    """    
    Takes a DataFrame, training, and validation data as its input.
    Returns f1-score, features and their coefficients, and predicted non-re-orders and re-orders.
    """
    
    clf = GaussianNB(var_smoothing=1e-9)
    clf.fit(X_tr, y_tr)
    vals = pd.DataFrame(clf.predict(X_val))[0].value_counts()
    print('Our f1-score is',f1_score(clf.predict(X_val), y_val))
    print('And we\'ve predicted',vals[0],'non-re-orders and',
    vals[1],'re-orders.')
    
    
    
    
