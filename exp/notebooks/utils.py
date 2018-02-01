from imblearn.over_sampling import SMOTE
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from tqdm import tqdm
from utils import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sc
import scipy.stats as ss
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
    return - sum( (a / (sum(a) + 1)) * np.log((a / (sum(a) + 1) + 1)))


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


def get_submission_preds(test_set, model, orig_cols, keep_cols):
    test_set = test_set[keep_cols]
    test_set = pre_process_data(test_set)
    #TODO: testing...
    test_set = get_distance(test_set, metric='hamming')
    test_set = get_distance(test_set, metric='euclidean')
    
    # Delete new columns that were not in training set
    diff = set(test_set.columns.tolist()) - set(orig_cols)
    
    test_set = test_set[test_set.columns.difference(list(diff))]
    
    # Add dummy columns that are not in the test set
    diff = set(orig_cols) - set(test_set.columns.tolist())
    test_set = test_set.assign(**{c: 0 for c in diff})
    
    # Reorder columns in the original way so XGBoost does not explode
    test_set = test_set[orig_cols]
    
    test_set.fillna(0, inplace=True)
    
    testxgb = xgb.DMatrix(test_set)
    
    return  model.predict(testxgb)


def cramers_corrected_stat(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))


def get_highly_correlated_columns(df):
    df_objs = df.select_dtypes(include=['O'])
    #del df_objs['country']
    corr_matrix = pd.DataFrame(columns=df_objs.columns, index=df_objs.columns.tolist())
    with tqdm(total=len(corr_matrix.columns.tolist()) * len(corr_matrix.columns.tolist())) as pbar:
        for col1 in df_objs.columns.tolist():
            for col2 in df_objs.columns.tolist():
                if col1 != col2:
                    confusion_matrix = pd.crosstab(df_objs[col1], df_objs[col2])
                    corr = cramers_corrected_stat(confusion_matrix)
                else:
                    corr = 1
                corr_matrix.loc[col1, col2] = corr
                pbar.update(1)
    
    cols = {}
    for c1 in corr_matrix.columns.tolist():
        s = corr_matrix.loc[c1]
        s = s[s > 0.5]
        s = list(s.index)
        s.remove(c1)
        cols[c1] = s

    cols = {k: cols[k] for k in cols if cols[k]}
    return cols


def categorize_continous(df, num_cats=3):
    df_nums = df.select_dtypes(include=['int64', 'float64'])
    for col in df_nums.columns.tolist():
        df['{}_{}cats'.format(col, num_cats)] = pd.cut(df_nums[col], num_cats, labels=[str(e) for e in range(num_cats)])
    return df


def prepare_indiv_hhold_set(train_set_ind):
    # Lets categorize nums, lets standardize the nums then
    train_set_ind_nums = train_set_ind.select_dtypes(include=['int64', 'float64'])
    to_standar = train_set_ind_nums.columns.tolist()
    to_standar.remove('iid')
    train_set_ind_nums.fillna(0, inplace=True) 
    train_set_ind_nums = categorize_continous(train_set_ind_nums.drop('iid', axis=1), num_cats=4)
    # TODO: this is being normalized twice (one more time after I merge this with the main dataset
    train_set_ind_nums[to_standar] = (train_set_ind_nums[to_standar] - train_set_ind_nums[to_standar].mean()) / train_set_ind_nums[to_standar].std()
    train_set_ind[train_set_ind_nums.columns] = train_set_ind_nums
    # Create new column with family members count
    train_set_ind['fam_count'] = train_set_ind['iid'].groupby(train_set_ind['iid'].index.get_level_values(0)).count()
    #train_set_ind['fam_count'] = train_set_ind.fam_count.astype('category')
    # Delete redundant columns (its information has already been encoded in other columns)
    del train_set_ind['iid']
    del train_set_ind['country']
    #if 'poor' in train_set_ind.columns.tolist():
    #    del train_set_ind['poor']
    return train_set_ind


def get_distance(df, metric='hamming'):
    dist_func = {
        'hamming': hamming_dist,
        'euclidean': euclidean_dist
    }
    only_bool_cols = [col for col in df.columns.tolist() if '_' in col]
    indexes = list(set(df.index.get_level_values(0)))
    avgs = []
    medians = []
    stds = []
    for i in indexes:
        tmp_df = df[df.index == i]
        dist_matrix = dist_func[metric](tmp_df)
        dist_matrix = np.array(
                [dist_matrix[k, j] for k in range(dist_matrix.shape[0]) for j in range(dist_matrix.shape[1]) if k!=j ])

        # Lets count the decimals only
        if metric == 'hamming':
            dist_matrix = 100 * (1 - dist_matrix)

        # TODO: try avg/std/median of each individual w.r.t.t.o instead of the whole household avg/std/median
        avg = np.mean(dist_matrix)
        std = np.std(dist_matrix)
        median = np.median(dist_matrix)

        avgs.extend([avg] * len(tmp_df))
        stds.extend([std] * len(tmp_df))
        medians.extend([median] * len(tmp_df))

    df[metric+'_mean'] = avgs
    df[metric+'_std'] = stds 
    df[metric+'_median'] = medians
    return df 

def hamming_dist(df):
    return 1 - pairwise_distances(df, metric = "hamming") 

def euclidean_dist(df):
    distances = pdist(df, metric='euclidean')
    return squareform(distances)
