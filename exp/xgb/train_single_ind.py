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

def learning_rates(boosting_round, num_boost_round):
    if boosting_round < 2000:
        return 0.001
    if boosting_round > 2000 and boosting_round < 2300:
        return 0.001
    else:
        return 0.005


## 1 - PREPARE DATASETS
train_set = pd.read_csv(DATA_PATHS[COUNTRY]['train'], index_col='id')
train_set_ind = pd.read_csv(DATA_PATHS_IND[COUNTRY]['train'], index_col='id')
train_set_ind = prepare_indiv_hhold_set(train_set_ind)

train_set = train_set.merge(train_set_ind, how='left', left_index=True, right_index=True)

# We need to repreprocess the data with less columns
print("Country {}".format(COUNTRY))
# Filter out low entropy columns
train_reduc = filter_columns(train_set.drop('poor', axis=1))
# Normalize numeric columns, create dummies
X_train = pre_process_data(train_reduc)
# Create labels
train_set.poor.fillna(False, inplace=True)
y_train = np.ravel(train_set.poor.astype(int))

test_size = 0.2
xgb_x_train, xgb_x_test, xgb_y_train, xgb_y_test = prepare_data(X_train, y_train, test_size=test_size, xgb_format=True)


# 2 - HYPERPARAMETERS OPT 
num_round = 15000 
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
    'seed': 42
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
#}


early_stopping = 100 
xgb_model = xgb.train(params, xgb_x_train, num_round, evals=[(xgb_x_train, 'train_set'), (xgb_x_test, 'test_set')], early_stopping_rounds=early_stopping, learning_rates=learning_rates, verbose_eval=True)

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
test_set_ind = pd.read_csv(DATA_PATHS_IND[COUNTRY]['test'], index_col='id')
test_set_ind = prepare_indiv_hhold_set(test_set_ind)
test_set = test_set.merge(test_set_ind, how='left', left_index=True, right_index=True)

# Get predictions
preds = get_submission_preds(test_set, xgb_model, X_train.columns.tolist(), train_reduc.columns.tolist())

# Clean results
sub = make_country_sub(preds, test_set, COUNTRY)
grouped_sub = sub.groupby(sub.index.get_level_values(0)).mean()
grouped_sub = pd.DataFrame(sub)
grouped_sub['country'] = 'A'

#TODO: this line is unnecessary
submission = pd.concat([grouped_sub])
submission.to_csv('submission_recent_XGB_indiv_{}.csv'.format(COUNTRY))

print("Submission prepared. Saving models")
xgb_model.save_model(str(MODELS_DIR / 'xgb_{}_indiv.xgb'.format(COUNTRY)))
print("Models saved. Good luck!")
