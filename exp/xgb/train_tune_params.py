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
    if boosting_round < 500:
        return 0.01
    if boosting_round > 500 and boosting_round < 1000:
        return 0.005
    else:
        return 0.002


# Lets try to find the optimal hyperparameters for each country
def tune_params(dtrain, dtest, y_test):
    params = {
        'max_depth': 10,
        'eta': 0.05, 
        'silent': 1,
        'lambda': 0.5,
        'alpha': 0.5,
        'lambda_bias': 0.5, 
        'min_child_weight': 1,
        'objective': 'binary:logistic', 
        'eval_metric': 'logloss', 
        'seed': 42,
        'tree_method': 'gpu_exact'
    }

    current_loss = 10000
    current_rocauc = 0
    model = None
    best_num_rounds = 0
    best_hyperparams = {}
    num_rounds = [2000]
    max_depths = [2, 5, 10]
    etas = [0.005, 0.01]
    min_child_weights = [1, 2]
    lamdas = [0.2, 0.5, 0.8, 1]
    alphas = [0.2, 0.5, 0.8, 1]
    lambda_biases = [0.5, 0.8, 1]
    total_combinations = len(num_rounds) * len(max_depths) * len(etas)*\
        len(min_child_weights) * len(lamdas) * len(alphas) * len(lambda_biases)

    with tqdm(total=total_combinations) as pbar:
        for num_round in num_rounds:
            for max_depth in max_depths:
                for eta in etas:
                    for min_child_weight in min_child_weights:
                        for lamda in lamdas:
                            for alpha in alphas:
                                for lambda_bias in lambda_biases:
                                    params['max_depth'] = max_depth
                                    params['eta'] = eta
                                    params['min_child_weight'] = min_child_weight
                                    params['lambda'] = lamda
                                    params['alpha'] = alpha
                                    params['lambda_bias'] = lambda_bias

                                    model = train_xgb_model(dtrain, params=params, num_round=num_round)
                                    model = xgb.train(params, dtrain, num_round, evals=[(dtest, 'test_set')], verbose_eval=False)

                                    pred = model.predict(dtest)
                                    rocauc = roc_auc_score(y_test, pred)
                                    loss = float((model.eval(dtest)).split('logloss:')[1])

                                    if rocauc > current_rocauc:
                                        current_rocauc = rocauc
                                        current_loss = loss
                                        best_hyperparams = params
                                        best_num_rounds = num_round
                                        best_model = model
                                    pbar.update(1)

    return best_model, best_hyperparams, best_num_rounds, current_loss, current_rocauc


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
xgb_model, best_params, num_rounds, loss, rocauc = tune_params(xgb_x_train, xgb_x_test, xgb_y_test)

print("\n\nCountry {} best Hyperparams with num_round = {}:\n\n".format(COUNTRY, num_rounds))
print(best_params)
print("Results: Loss: {} -- ROC AUC: {}".format(loss, rocauc))

## 4 - PREPARE SUBMISSION
test_set = pd.read_csv(DATA_PATHS[COUNTRY]['test'], index_col='id')

preds = get_submission_preds(test_set, xgb_model, X_train.columns.tolist(), train_reduc.columns.tolist())

sub = make_country_sub(preds, test_set, COUNTRY)

submission = pd.concat([sub])
submission.to_csv('submission_recent_XGB_tuned_{}.csv'.format(COUNTRY))

print("Submission prepared. Saving models")
xgb_model.save_model(str(MODELS_DIR / 'xgb_{}_tuned.xgb'.format(COUNTRY)))
print("Models saved. Good luck!")
