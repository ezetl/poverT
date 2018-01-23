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


## 2 - HYPERPARAMETERS OPT 
#num_round = 10000 
#params = {
#    'max_depth': 15,
#    'eta': 0.005,
#    'silent': 1,
#    'lambda': 1,
#                                         #                     test     train          test     train
#                                         # A: Gamma 5: ROC 0.947926  0.961984 LOSS 0.297549  0.265718   (para la proxima usar 20K iteraciones asi sigue bajando)
#    'gamma': 5,                          # B: Gamma 5: ROC 0.848453  0.887216 LOSS 0.216402  0.204157   (10K iteraciones)
#                                         # C: Gamma 5: ROC 0.997975  0.99878  LOSS 0.03478   0.030321   (10K iteraciones)
#    'alpha': 8,
#    'lambda_bias': 8,
#    'min_child_weight': 2,
#    'objective': 'binary:logistic',
#    'eval_metric': 'logloss',
#    'subsample': 0.5,                    # Avoid overfitting
#    'colsample_bytree': 0.5,             # Avoid overfitting
#    'seed': 42
#}
num_round = 2500 
params = {
    'max_depth': 9,
    'eta': 0.01,
    'silent': 1,
    'lambda': 0.8,
    'gamma': 0,
    'alpha': 1,
    'lambda_bias': 1,
    'min_child_weight': 2,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42
}


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
test_set = pd.read_csv(DATA_PATHS[COUNTRY]['test'], index_col='id')

preds = get_submission_preds(test_set, xgb_model, X_train.columns.tolist(), train_reduc.columns.tolist())

sub = make_country_sub(preds, test_set, COUNTRY)

submission = pd.concat([sub])
submission.to_csv('submission_recent_XGB_tuned_{}.csv'.format(COUNTRY))

print("Submission prepared. Saving models")
xgb_model.save_model(str(MODELS_DIR / 'xgb_{}_tuned.xgb'.format(COUNTRY)))
print("Models saved. Good luck!")
