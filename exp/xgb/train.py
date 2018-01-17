#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
import sys
sys.path.append(str(Path.cwd() / '..' / 'notebooks'))
from utils import *


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

# We need to repreprocess the data with less columns
print("Country A")
# Filter out low entropy columns
a_train_reduc = filter_columns(a_train.drop('poor', axis=1))
# Normalize numeric columns, create dummies
aX_train = pre_process_data(a_train_reduc)
# Create labels
a_train.poor.fillna(False, inplace=True)
ay_train = np.ravel(a_train.poor.astype(int))

print("\nCountry B")
b_train_reduc = filter_columns(b_train.drop('poor', axis=1))
bX_train = pre_process_data(b_train_reduc)
b_train.poor.fillna(False, inplace=True)
by_train = np.ravel(b_train.poor.astype(int))

print("\nCountry C")
c_train_reduc = filter_columns(c_train.drop('poor', axis=1))
cX_train = pre_process_data(c_train_reduc)
c_train.poor.fillna(False, inplace=True)
cy_train = np.ravel(c_train.poor.astype(int))


test_size = 0.2
xgb_ax_train, xgb_ax_test, xgb_ay_train, xgb_ay_test = prepare_data(aX_train, ay_train, test_size=test_size, xgb_format=True)
xgb_bx_train, xgb_bx_test, xgb_by_train, xgb_by_test = prepare_data(bX_train, by_train, test_size=test_size, xgb_format=True)
xgb_cx_train, xgb_cx_test, xgb_cy_train, xgb_cy_test = prepare_data(cX_train, cy_train, test_size=test_size, xgb_format=True)


## 2 - TRAIN MODELS

num_round = 10
params = {'max_depth': 3, 'eta': 0.01, 'silent': 1, 'lambda': 0.8, 'alpha': 0.8, 'lambda_bias': 0.5, 'min_child_weight': 2, 'objective': 'binary:logistic', 'eval_metric': 'logloss', 'seed': 42}

early_stopping = 500
xgb_a = xgb.train(params, xgb_ax_train, num_round, evals=[(xgb_ax_train, 'a_train'), (xgb_ax_test, 'a_test')], early_stopping_rounds=early_stopping, learning_rates=learning_rates, verbose_eval=True)
xgb_b = xgb.train(params, xgb_bx_train, num_round, evals=[(xgb_bx_train, 'b_train'), (xgb_bx_test, 'b_test')], early_stopping_rounds=early_stopping, learning_rates=learning_rates, verbose_eval=True)
xgb_c = xgb.train(params, xgb_cx_train, num_round, evals=[(xgb_cx_train, 'c_train'), (xgb_cx_test, 'c_test')], early_stopping_rounds=early_stopping, learning_rates=learning_rates, verbose_eval=True)

## 3 - EVALUATE AND COMPUTE LOSSES

losses = {'test': {'A': 0, 'B': 0, 'C': 0}, 'train': {'A': 0, 'B': 0, 'C': 0}}
losses['test']['A'] = float((xgb_a.eval(xgb_ax_test)).split('logloss:')[1])
losses['test']['B'] = float((xgb_b.eval(xgb_bx_test)).split('logloss:')[1])
losses['test']['C'] = float((xgb_c.eval(xgb_cx_test)).split('logloss:')[1])

losses['train']['A'] = float((xgb_a.eval(xgb_ax_train)).split('logloss:')[1])
losses['train']['B'] = float((xgb_b.eval(xgb_bx_train)).split('logloss:')[1])
losses['train']['C'] = float((xgb_c.eval(xgb_cx_train)).split('logloss:')[1])

losses = pd.DataFrame.from_records(losses)
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

a_preds = get_submission_preds(a_test, xgb_a, aX_train.columns.tolist(), a_train_reduc.columns.tolist())
b_preds = get_submission_preds(b_test, xgb_b, bX_train.columns.tolist(), b_train_reduc.columns.tolist())
c_preds = get_submission_preds(c_test, xgb_c, cX_train.columns.tolist(), c_train_reduc.columns.tolist())

a_sub = make_country_sub(a_preds, a_test, 'A')
b_sub = make_country_sub(b_preds, b_test, 'B')
c_sub = make_country_sub(c_preds, c_test, 'C')

submission = pd.concat([a_sub, b_sub, c_sub])
submission.to_csv('submission_recent_XGB_best.csv')
