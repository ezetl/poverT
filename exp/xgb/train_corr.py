#!/usr/bin/env python3
from pathlib import Path
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
import sys
sys.path.append(str(Path.cwd() / '..' / 'notebooks'))
from utils import *


MODELS_DIR = Path.cwd() / 'models'

def learning_rates(boosting_round, num_boost_round):
    if boosting_round < 500:
        return 0.01
    if boosting_round > 500 and boosting_round < 1000:
        return 0.005
    else:
        return 0.002


## 1 - PREPARE DATASETS
a_train = pd.read_csv(DATA_PATHS['A']['train'], index_col='id')
b_train = pd.read_csv(DATA_PATHS['B']['train'], index_col='id')
c_train = pd.read_csv(DATA_PATHS['C']['train'], index_col='id')

def delete_columns(df, dict_cols):
    to_delete = list(set([elem for values in dict_cols.values() for elem in values]))
    return df.drop(to_delete, axis=1)

# We need to repreprocess the data with less columns
print("Country A")
# Filter out low entropy columns
a_train = balance_up_down(a_train)
a_train_reduc = a_train
#a_train_reduc = filter_columns(a_train.drop('poor', axis=1))
a_corr_cols = get_highly_correlated_columns(a_train_reduc)
a_train_small = delete_columns(a_train_reduc, a_corr_cols)
# Normalize numeric columns, create dummies
if 'poor' in a_train_small.columns.tolist():
    del a_train_small['poor']
aX_train = pre_process_data(a_train_small)
# Create labels
a_train.poor.fillna(False, inplace=True)
ay_train = np.ravel(a_train.poor.astype(int))

print("\nCountry B")
b_train = balance_up_down(b_train)
b_train_reduc = b_train
#b_train_reduc = filter_columns(b_train.drop('poor', axis=1))
b_corr_cols = get_highly_correlated_columns(b_train_reduc)
b_train_small = delete_columns(b_train_reduc, b_corr_cols)
if 'poor' in b_train_small.columns.tolist():
    del b_train_small['poor']
bX_train = pre_process_data(b_train_small)
b_train.poor.fillna(False, inplace=True)
by_train = np.ravel(b_train.poor.astype(int))

print("\nCountry C")
c_train = balance_up_down(c_train)
c_train_reduc = c_train
#c_train_reduc = filter_columns(c_train.drop('poor', axis=1))
c_corr_cols = get_highly_correlated_columns(c_train_reduc)
c_train_small = delete_columns(c_train_reduc, c_corr_cols)
if 'poor' in c_train_small.columns.tolist():
    del c_train_small['poor']
cX_train = pre_process_data(c_train_small)
c_train.poor.fillna(False, inplace=True)
cy_train = np.ravel(c_train.poor.astype(int))


test_size = 0.2
xgb_ax_train, xgb_ax_test, xgb_ay_train, xgb_ay_test = prepare_data(aX_train, ay_train, test_size=test_size, xgb_format=True)
xgb_bx_train, xgb_bx_test, xgb_by_train, xgb_by_test = prepare_data(bX_train, by_train, test_size=test_size, xgb_format=True)
xgb_cx_train, xgb_cx_test, xgb_cy_train, xgb_cy_test = prepare_data(cX_train, cy_train, test_size=test_size, xgb_format=True)


## 2 - TRAIN MODELS

num_round = 3000
params = {'max_depth': 3, 'eta': 0.01, 'silent': 1, 'lambda': 0.8, 'alpha': 1, 'lambda_bias': 0.5, 'min_child_weight': 2, 'objective': 'binary:logistic', 'eval_metric': 'logloss', 'seed': 42, 'tree_method': 'gpu_exact'}

early_stopping = 300
xgb_a = xgb.train(params, xgb_ax_train, num_round, evals=[(xgb_ax_test, 'a_test')], early_stopping_rounds=early_stopping, learning_rates=learning_rates, verbose_eval=True)
xgb_b = xgb.train(params, xgb_bx_train, num_round, evals=[(xgb_bx_test, 'b_test')], early_stopping_rounds=early_stopping, learning_rates=learning_rates, verbose_eval=True)
xgb_c = xgb.train(params, xgb_cx_train, num_round, evals=[(xgb_cx_test, 'c_test')], early_stopping_rounds=early_stopping, learning_rates=learning_rates, verbose_eval=True)

## 3 - EVALUATE AND COMPUTE LOSSES

a_test_res = xgb_a.predict(xgb_ax_test)
b_test_res = xgb_b.predict(xgb_bx_test)
c_test_res = xgb_c.predict(xgb_cx_test)
a_train_res = xgb_a.predict(xgb_ax_train)
b_train_res = xgb_b.predict(xgb_bx_train)
c_train_res = xgb_c.predict(xgb_cx_train)

rocaucs = {'test': {'A': 0, 'B': 0, 'C': 0}, 'train': {'A': 0, 'B': 0, 'C': 0}}
rocaucs['test']['A'] = roc_auc_score(xgb_ay_test, a_test_res)
rocaucs['test']['B'] = roc_auc_score(xgb_by_test, b_test_res)
rocaucs['test']['C'] = roc_auc_score(xgb_cy_test, c_test_res)
rocaucs['train']['A'] = roc_auc_score(xgb_ay_train, a_train_res)
rocaucs['train']['B'] = roc_auc_score(xgb_by_train, b_train_res)
rocaucs['train']['C'] = roc_auc_score(xgb_cy_train, c_train_res)
rocaucs = pd.DataFrame.from_records(rocaucs)
print("ROC AUC Results")
print(rocaucs)

losses = {'test': {'A': 0, 'B': 0, 'C': 0}, 'train': {'A': 0, 'B': 0, 'C': 0}}
losses['test']['A'] = float((xgb_a.eval(xgb_ax_test)).split('logloss:')[1])
losses['test']['B'] = float((xgb_b.eval(xgb_bx_test)).split('logloss:')[1])
losses['test']['C'] = float((xgb_c.eval(xgb_cx_test)).split('logloss:')[1])
losses['train']['A'] = float((xgb_a.eval(xgb_ax_train)).split('logloss:')[1])
losses['train']['B'] = float((xgb_b.eval(xgb_bx_train)).split('logloss:')[1])
losses['train']['C'] = float((xgb_c.eval(xgb_cx_train)).split('logloss:')[1])

losses = pd.DataFrame.from_records(losses)
print("Loss Results")
print(losses)

lines = sum([len(aX_train), len(bX_train), len(cX_train)])
weights = [len(aX_train) / lines, len(bX_train) / lines, len(cX_train) / lines]
total_test_loss = np.average(losses['test'], weights=weights)
total_train_loss = np.average(losses['train'], weights=weights)
print("Averaged test loss: {}".format(total_test_loss))
print("Averaged train loss: {}".format(total_train_loss))


## 4 - PREPARE SUBMISSION
a_test = pd.read_csv(DATA_PATHS['A']['test'], index_col='id')
b_test = pd.read_csv(DATA_PATHS['B']['test'], index_col='id')
c_test = pd.read_csv(DATA_PATHS['C']['test'], index_col='id')

a_preds = get_submission_preds(a_test, xgb_a, aX_train.columns.tolist(), a_train_small.columns.tolist())
b_preds = get_submission_preds(b_test, xgb_b, bX_train.columns.tolist(), b_train_small.columns.tolist())
c_preds = get_submission_preds(c_test, xgb_c, cX_train.columns.tolist(), c_train_small.columns.tolist())

a_sub = make_country_sub(a_preds, a_test, 'A')
b_sub = make_country_sub(b_preds, b_test, 'B')
c_sub = make_country_sub(c_preds, c_test, 'C')

submission = pd.concat([a_sub, b_sub, c_sub])
submission.to_csv('submission_recent_XGB_best1.csv')

print("Submission prepared. Saving models")
xgb_a.save_model(str(MODELS_DIR / 'xgb_a.xgb'))
xgb_b.save_model(str(MODELS_DIR / 'xgb_b.xgb'))
xgb_c.save_model(str(MODELS_DIR / 'xgb_c.xgb'))
print("Models saved. Good luck!")
