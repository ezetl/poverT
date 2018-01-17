from imblearn.over_sampling import SMOTE
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from utils import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sc
import xgboost as xgb


DATA_DIR = Path.cwd() / '..' / '..' / 'data'
DATA_PATHS = {
    'A': {
        'train': str(DATA_DIR / 'A_hhold_train.csv'), 
        'test': str(DATA_DIR / 'A_hhold_test.csv')
    }, 
    'B': {
        'train': str(DATA_DIR / 'B_hhold_train.csv'), 
        'test': str(DATA_DIR / 'B_hhold_test.csv')
    },
    'C': {
        'train': str(DATA_DIR / 'C_hhold_train.csv'), 
        'test': str(DATA_DIR / 'C_hhold_test.csv')
    }
}

DATA_PATHS_IND = {
    'A': {
        'train': str(DATA_DIR / 'A_indiv_train.csv'), 
        'test': str(DATA_DIR / 'A_indiv_test.csv')
    }, 
    'B': {
        'train': str(DATA_DIR / 'B_indiv_train.csv'), 
        'test': str(DATA_DIR / 'B_indiv_test.csv')
    },
    'C': {
        'train': str(DATA_DIR / 'C_indiv_train.csv'), 
        'test': str(DATA_DIR / 'C_indiv_test.csv')
    }
}


# Standardize features
def standardize(df, numeric_only=True):
    numeric = df.select_dtypes(include=['int64', 'float64'])
    
    # subtract mean and divide by std
    df[numeric.columns] = (numeric - numeric.mean()) / numeric.std()
    
    return df
    

def pre_process_data(df, enforce_cols=None):
    print("Input shape:\t{}".format(df.shape))
        

    df = standardize(df)
    print("After standardization {}".format(df.shape))
        
    # create dummy variables for categoricals
    df = pd.get_dummies(df)
    print("After converting categoricals:\t{}".format(df.shape))
    

    # match test set and training set columns
    if enforce_cols is not None:
        to_drop = np.setdiff1d(df.columns, enforce_cols)
        to_add = np.setdiff1d(enforce_cols, df.columns)

        df.drop(to_drop, axis=1, inplace=True)
        df = df.assign(**{c: 0 for c in to_add})
    
    df.fillna(0, inplace=True)

    return df

def entropy(a):
    return - sum( (a / sum(a)) * np.log((a / sum(a))))


def get_entropies(df):
    entropies = []
    for col in df.columns.tolist():
        res = df[col].value_counts()
        entropies.append(entropy(res.values))

    return entropies


def get_low_entropy_columns(df):
    to_del = []
    entropies = get_entropies(df)
    median_entr = np.median(entropies)
    #std_entr = np.std(entropies)
    #avg_entr = np.mean(entropies)
    for i, col in enumerate(df.columns.tolist()):
        if entropies[i] < median_entr:
            to_del.append(col)
    return to_del


def filter_columns(df):
    to_del = get_low_entropy_columns(df)
    print("Total columns: {}. To delete: {}".format(len(df.columns.tolist()), len(to_del)))
    to_keep = set(df.columns.tolist()) - set(to_del)
    return df[list(to_keep)]


def make_country_sub(preds, test_feat, country):
    # make sure we code the country correctly
    country_codes = ['A', 'B', 'C']
    
    # get just the poor probabilities
    country_sub = pd.DataFrame(data=preds,
                               columns=['poor'], 
                               index=test_feat.index)

    
    # add the country code for joining later
    country_sub["country"] = country
    return country_sub[["country", "poor"]]


def prepare_data(x, y, test_size=0.2, xgb_format=False):
    if test_size == 0:
        dtrain = x
        Y_train = y
        dtest = None
        Y_test = None
    else:
        dtrain, dtest, Y_train, Y_test = train_test_split(x, y, test_size=test_size, stratify=y, random_state=42)

    if xgb_format:
        dtrain = xgb.DMatrix(dtrain, label=Y_train)
        if test_size:
            dtest = xgb.DMatrix(dtest, label=Y_test)

    return dtrain, dtest, Y_train, Y_test


def train_rf_model(features, labels, **kwargs):

    # instantiate model
    model = RandomForestClassifier(n_estimators=50, random_state=0)

    # train model
    model.fit(features, labels)

    # get a (not-very-useful) sense of performance
    accuracy = model.score(features, labels)
    print(f"In-sample accuracy: {accuracy:0.2%}")

    return model


def train_xgb_model(dtrain, deval=None, params=None, num_round=100, early_stopping=500):
    if params is None:
        params = {'max_depth': 4, 'eta': 0.01, 'silent': 1, 'objective': 'reg:logistic'}

    if deval is not None:
        bst = xgb.train(params, dtrain, num_round, evals=deval, early_stopping_rounds=early_stopping, verbose_eval=True)
    else:
        bst = xgb.train(params, dtrain, num_round, verbose_eval=True)
    return bst


# Compute loss
# -log P(yt|yp) = -(yt log(yp) + (1 - yt) log(1 - yp))
def log_loss(yt, yp):
    # yt: groundtruth
    # yp: predicted
    ground = np.array(yt)
    pred = yp.astype(float)
    eps_pred = np.maximum(np.minimum(pred, 1. - 1e-15), 1e-15)
    loss = -(ground * np.log(eps_pred) + (1 - ground) * np.log(1 - eps_pred))
    return np.mean(loss)

# Cross Validate
def cross_validate(x_train, x_test, y_train, y_test, model):
    test_loss = None
    if x_test is not None:
        preds = model.predict(x_test)
        test_loss = log_loss(preds, y_test)

    preds_train = model.predict(x_train)
    train_loss = log_loss(preds_train, y_train)
    return train_loss, test_loss


def balance(df):
    poor = df[df['poor'] == True]
    not_poor = df[df['poor'] == False]
    poor_upsampled = resample(poor, 
                              replace=True,
                              n_samples=len(not_poor),
                              random_state=42)
    res = pd.concat([poor_upsampled, not_poor])
    return res.sample(frac=1)


def balance_up_down(df):
    poor = df[df['poor'] == True]
    not_poor = df[df['poor'] == False]
    
    not_poor_downsampled = resample(not_poor, 
                              replace=True,
                              n_samples=int(len(not_poor) * 0.6),
                              random_state=42)
    
    poor_upsampled = resample(poor, 
                              replace=True,
                              n_samples=len(not_poor_downsampled),
                              random_state=42)
    res = pd.concat([poor_upsampled, not_poor_downsampled])
    return res.sample(frac=1)
 

def balance_smote(x, y):
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_sample(x, y)
    return X_res, y_res



def encode_dataset(raw_df):
    df_bala = balance(raw_df)
    #df_bala = filter_columns(df_bala.drop('poor', axis=1))
    X_train = pre_process_data(df_bala)
    raw_df.poor.fillna(False, inplace=True)
    y_train = np.ravel(df_bala.poor.astype(int))
    return X_train, y_train


def get_preds(test_set, model, orig_cols, keep_cols):
    test_set = test_set[keep_cols]
    test_set = pre_process_data(test_set)
    
    # Delete new columns that were not in training set
    diff = set(test_set.columns.tolist()) - set(orig_cols)
    
    test_set = test_set[test_set.columns.difference(list(diff))]
    
    # Add dummy columns that are not in the test set
    diff = set(orig_cols) - set(test_set.columns.tolist())
    test_set = test_set.assign(**{c: 0 for c in a_diff})
    
    # Reorder columns in the original way so XGBoost does not explode
    test_set = test_set[orig_cols]
    
    test_set.fillna(0, inplace=True)
    
    testxgb = xgb.DMatrix(test_set)
    
    return  model.predict(testxgb)
