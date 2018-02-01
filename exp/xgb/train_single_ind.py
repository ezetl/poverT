#!/usr/bin/env python3
from pathlib import Path
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import numpy as np
import pandas as pd
import sys
sys.path.append(str(Path.cwd() / '..' / 'notebooks'))
from utils import *


MODELS_DIR = Path.cwd() / 'models'
COUNTRY = sys.argv[1] 


# Keep columns from individual data datasets. Obtained from notebook 8
if COUNTRY == 'A':
    keep_indiv_cols = ['iid', 'gtnNTNam', 'OdXpbPGJ', 'XONDGWjH', 'FPQrjGnS', 'UsmeXdIS', 'igHwZsYz', 'AoLwmlEH', 'QvgxCmCV', 'ukWqmeSS', 'poor', 'country']
    keep_hhold_cols = ['SlDKnCuu', 'DsKacCdL', 'rtPrBBPl', 'jdetlNNF', 'TYhoEiNm', 'nGTepfos', 'DxLvCGgv', 'uSKnVaKV', 'BbKZUYsB', 'UCnazcxd', 'EfkPrfXa', 'nEsgxvAq', 'NmAVTtfA', 'YTdCRVJt', 'QyBloWXZ', 'ZRrposmO', 'IIEHQNUc', 'HfKRIwMb', 'NRVuZwXK', 'UCAmikjV', 'UGbBCHRE', 'uJYGhXqG', 'ltcNxFzI', 'ggNglVqE', 'JwtIxvKg', 'FlBqizNL', 'bEPKkJXP', 'cqUmYeAp', 'tZKoAqgl', 'TqrXZaOw', 'galsfNtg', 'ihGjxdDj', 'gwhBRami', 'bPOwgKnT', 'YWwNfVtR', 'wxDnGIwN', 'bMudmjzJ', 'OnTaJkLa', 'OMtioXZZ', 'LjvKYNON', 'wwfmpuWA', 'znHDEHZP', 'HHAeIHna', 'CrfscGZl', 'dCGNTMiG', 'ngwuvaCV', 'GnUDarun', 'NanLCXEI', 'ZnBLVaqz', 'lQQeVmCa', 'lFcfBRGd', 'wEbmsuJO', 'pWyRKfsb', 'ErggjCIN', 'IZFarbPw', 'YFMZwKrU', 'uizuNzbk', 'dlyiMEQt', 'GhJKwVWC', 'lVHmBCmb', 'EuJrVjyG', 'CpqWSQcW', 'jxSUvflR', 'eoNxXdlZ', 'qgxmqJKa', 'gfurxECf', 'CbzSWtkF', 'XDDOZFWf', 'CIGUXrRQ', 'QayGNSmS', 'ePtrWTFd', 'tbsBPHFD', 'naDKOzdk', 'DNAfxPzs', 'xkUFKUoW', 'dEpQghsA', 'jVDpuAmP', 'SeZULMCT', 'AtGRGAYi', 'rYvVKPAF', 'NBfffJUe', 'mvgxfsRb', 'KHzKOKPw', 'UXfyiodk', 'mycoyYwl', 'BfGjiYom', 'iWEFJYkR', 'ogHwwdzc', 'BCehjxAl', 'nqndbwXP', 'phwExnuQ', 'CNkSTLvx', 'pjHvJhoZ', 'xZBEXWPR', 'bCYWWTxH', 'EQKKRGkR', 'ItpCDLDM', 'gOGWzlYC', 'ptEAnCSs', 'orfSPOJX', 'QBJeqwPF', 'XVwajTfe', 'jwEuQQve', 'kLkPtNnh', 'DbUNVFwv', 'FmSlImli', 'TiwRslOh', 'PWShFLnY', 'uRFXnNKV', 'lFExzVaF', 'tlxXCDiW', 'IKqsuNvV', 'uVnApIlJ', 'ktBqxSwa', 'GIMIxlmv', 'ncjMNgfp', 'UaXLYMMh', 'vRIvQXtC', 'WAFKMNwv', 'ZzUrQSMj', 'QZiSWCCB', 'LrDrWRjC', 'JCDeZBXq', 'AlDbXTlZ', 'poor', 'country']
elif COUNTRY == 'B':
    keep_indiv_cols = ['iid', 'TJGiunYp', 'esHWAAyG', 'fwRLKHlN', 'dnmwvCng', 'nkxrhykC', 'uDVGRBVU', 'wJthinfa', 'ulQCDoYe', 'poor', 'country']
    keep_hhold_cols = ['wJthinfa', 'euTESpHe', 'hQDJpUTd', 'ctmENvnX', 'zMxwwVGT', 'VQMXmqDx', 'vuQrLzvK', 'iTXaBYWz', 'UDYHmGPq', 'wZoTauKG', 'QHJMESPn', 'MEmWXiUy', 'PIUliveV', 'ErXfvfyP', 'qrOrXLPM', 'BnmJlaKE', 'pChByymn', 'JmfFCszC', 'mPWHlBwK', 'xhxyrqCY', 'rgelGqck', 'PrSsgpNa', 'uHXkmVcG', 'qNrUWhsv', 'UCdxjZfA', 'jbpJuASm', 'kUGedOja', 'NYaVxhbI', 'rGXlOcWw', 'vZbYxaoB', 'uaHtjcqx', 'sGJAZEeR', 'uzNDcOYr', 'HvnEuEBI', 'utlAPPgH', 'xFMGVEam', 'IYZKvELr', 'yXpuYjeX', 'pInyVRtW', 'zkbvzkPn', 'TLqHuNIQ', 'VelOAjzj', 'BITMVzqW', 'BEyCyEUG', 'zBVfTPxZ', 'RcpCILQM', 'kYVdGKjZ', 'bPeUrCFr', 'uPOlDdSA', 'SwfwjbRf', 'OBRIToAY', 'qIqGCKGX', 'DGcwVOVy', 'gmjAuMKF', 'RUftVwTl', 'fyQTkTme', 'FZxzBDxm', 'LgAQBTzu', 'VvnxIDll', 'OdLduMEH', 'VyHofjLM', 'EEIzsjsu', 'GrLBZowF', 'LwqzULbf', 'XzxOZkAn', 'wRArirvZ', 'KNUpIgTJ', 'RLvvlQVW', 'ubrhKvOP', 'VfPWMKeX', 'nzSoWngR', 'iJhxdRrO', 'dkBXXyXU', 'mpIAZMUq', 'papNAyVA', 'nrLstcxr', 'aLTViWPH', 'vmLrLHUf', 'nKHmqfHF', 'sClXNjye', 'KQlBXFOa', 'TbDUmaHA', 'gKUsAWph', 'QcBOtphS', 'bJtNuLls', 'tSSwwSLI', 'ZMzkZIxG', 'fowptmNG', 'ciJQedKc', 'LhdvvAcC', 'shoeXCtj', 'bmlzNlAT', 'OGjOCVTC', 'gnCdSMVe', 'AZVtosGB', 'toZzckhe', 'BkiXyuSp', 'ChbSWYhO', 'poor', 'country']
elif COUNTRY == 'C':
    keep_indiv_cols = ['iid', 'AOSWkWKB', 'XKQWlRjk', 'tOfmAJyI', 'XKyOwsRR', 'CgAkQtOd', 'gxCmCVUk', 'qGqYslGF', 'ShCKQiAy', 'VGJlUgVG', 'poor', 'country']
    keep_hhold_cols = ['GRGAYimk', 'vmKoAlVH', 'LhUIIEHQ', 'DTNyjXJp', 'ABnhybHK', 'yiuxBjHP', 'KIUzCiTC', 'jmsRIiqp', 'lPMcVCxU', 'gZWEypOM', 'ueeRzZmV', 'eTYScDpy', 'FlsGEbwx', 'znHDEHZP', 'kLAQgdly', 'mQWlSyDC', 'KWZxwpOn', 'WWuPOkor', 'fHbDHQYU', 'TusIlNXO', 'GIwNbAsH', 'qLDzvjiU', 'lGhmlsZv', 'izNLFWMH', 'tXjyOtiS', 'dgGsziwz', 'dnLwwdCv', 'FmHiHbuZ', 'Bknkgmhs', 'wlNGOnRd', 'cmjTMVrd', 'oniXaptE', 'PuxuSSrt', 'ubefFytO', 'RGdWBZrK', 'BBPluVrb', 'IRMacrkM', 'vuhCltpc', 'coFdvtHB', 'xVfQXPkP', 'vsdZwVFE', 'XYfcLBql', 'EQSmcscG', 'pQGrypBw', 'DBjxSUvf', 'kiAJBGqv', 'wcNjwEuQ', 'aFKPYcDt', 'gAZloxqF', 'phbxKGlB', 'nTaJkLaJ', 'YACFXGNR', 'HNRJQbcm', 'RvKNoECV', 'ZZGQNLOX', 'hZMlrJCa', 'KnVaKVhK', 'tanBXtgi', 'snkiwkvf', 'tIeYAnmc', 'LwKcZtLN', 'HsxIHBAf', 'POJXrpmn', 'zaJvluwo', 'nRXRObKS', 'FnRNhlrr', 'DMslsIBE', 'vSqQCatY', 'bFEsoTgJ', 'obIQUcpS', 'eqJPmiPb', 'mmoCpqWS', 'poor', 'country']
else:
    raise Exception('Country must be A, B, or C. You provided {}'.format(COUNTRY))


def learning_rates(boosting_round, num_boost_round):
    if boosting_round < 2000:
        return 0.001
    if boosting_round > 2000 and boosting_round < 2300:
        return 0.001
    else:
        return 0.005


## 1 - PREPARE DATASETS
train_set = pd.read_csv(DATA_PATHS[COUNTRY]['train'], index_col='id')
#train_set = train_set[keep_hhold_cols]

train_set_ind = pd.read_csv(DATA_PATHS_IND[COUNTRY]['train'], index_col='id')
train_set_ind = train_set_ind[keep_indiv_cols]
# Create final labels
train_set_ind.poor.fillna(False, inplace=True)
y_train = np.ravel(train_set_ind.poor.astype(int))
train_set_ind = prepare_indiv_hhold_set(train_set_ind)

# We need to repreprocess the data with less columns
print("Country {}".format(COUNTRY))
# Filter out low entropy columns
train_reduc = filter_columns(train_set.drop('poor', axis=1))


train_reduc = train_reduc.merge(train_set_ind, how='left', left_index=True, right_index=True)
# We don't want repeated rows
train_reduc.drop_duplicates(inplace=True)


# Normalize numeric columns, create dummies
X_train = pre_process_data(train_reduc.drop(['poor'], axis=1))
X_train = get_distance(X_train, metric='hamming')
X_train = get_distance(X_train, metric='euclidean')

train_reduc.poor.fillna(False, inplace=True)
y_train = np.ravel(train_reduc.poor.astype(int))
del train_reduc['poor']

test_size = 0.2
xgb_x_train, xgb_x_test, xgb_y_train, xgb_y_test = prepare_data(X_train, y_train, test_size=test_size, xgb_format=True)


# 2 - HYPERPARAMETERS OPT 
num_round = 7000 
params = {
    'max_depth': 15,
    'eta': 0.005,
    'silent': 1,
    'lambda': 1,
    'gamma': 5,
    'alpha': 8,
    'lambda_bias': 8,
    'min_child_weight': 2,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42,
    'tree_method': 'gpu_exact'
}

## TEST HYPERPARAMS FOR FASTER RESULTS
#num_round = 2 
#params = {
#    'max_depth': 9,
#    'eta': 0.01,
#    'silent': 1,
#    'lambda': 0.8,
#    'gamma': 0,
#    'alpha': 1,
#    'lambda_bias': 1,
#    'min_child_weight': 2,
#    'objective': 'binary:logistic',
#    'eval_metric': 'logloss',
#    'subsample': 0.8,
#    'colsample_bytree': 0.8,
#    'seed': 42
#    'tree_method': 'gpu_exact'
#}


early_stopping = 100
# evals=[(xgb_x_train, 'train_set'), (xgb_x_test, 'test_set')], early_stopping_rounds=early_stopping
xgb_model = xgb.train(params, xgb_x_train, num_round, evals=[(xgb_x_train, 'train_set'), (xgb_x_test, 'test_set')], early_stopping_rounds=early_stopping, learning_rates=learning_rates, verbose_eval=100)

test_res = xgb_model.predict(xgb_x_test)
train_res = xgb_model.predict(xgb_x_train)
rocaucs = {'test': {COUNTRY: 0}, 'train': {COUNTRY: 0}}
rocaucs['test'][COUNTRY] = roc_auc_score(xgb_y_test, test_res)
rocaucs['train'][COUNTRY] = roc_auc_score(xgb_y_train, train_res)
rocaucs = pd.DataFrame.from_records(rocaucs)
print("ROC AUC Results")
print(rocaucs)

losses = {'test': {COUNTRY: 0}, 'train': {COUNTRY: 0}}
losses['test'][COUNTRY] = float((xgb_model.eval(xgb_x_test)).split('logloss:')[1])
losses['train'][COUNTRY] = float((xgb_model.eval(xgb_x_train)).split('logloss:')[1])
losses = pd.DataFrame.from_records(losses)
print("Loss Results")
print(losses)


## 4 - PREPARE SUBMISSION
# Load and prepare csvs
test_set = pd.read_csv(DATA_PATHS[COUNTRY]['test'], index_col='id')
indexes = test_set.index.get_level_values(0)
test_set_ind = pd.read_csv(DATA_PATHS_IND[COUNTRY]['test'], index_col='id')
test_set_ind = prepare_indiv_hhold_set(test_set_ind)
test_set = test_set.merge(test_set_ind, how='left', left_index=True, right_index=True)

# Get predictions
preds = get_submission_preds(test_set, xgb_model, X_train.columns.tolist(), train_reduc.columns.tolist())

# Clean results
sub = make_country_sub(preds, test_set, COUNTRY)
grouped_sub = sub.groupby(sub.index.get_level_values(0)).mean()
submission = pd.DataFrame(grouped_sub)
submission['country'] = COUNTRY 
# Reorder columns
submission = submission[['country', 'poor']]
# Reorder indexes
submission = submission.reindex(indexes)

submission.to_csv('submission_recent_XGB_indiv_{}.csv'.format(COUNTRY))

print("Submission prepared. Saving models")
xgb_model.save_model(str(MODELS_DIR / 'xgb_{}_indiv.xgb'.format(COUNTRY)))
print("Models saved. Good luck!")
