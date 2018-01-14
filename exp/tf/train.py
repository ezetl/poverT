#!/usr/bin/env python3
from pathlib import Path
from tf_utils import *
import pandas as pd
import tensorflow as tf
# export PYTHONPATH=~/Software/xgboost/python-package<Paste>


MODEL_DIR = Path.cwd() / 'models'
LOGS_DIR = Path.cwd() / 'logs'


# Load and prepare training data
a = pd.read_csv(DATA_PATHS['A']['train'], index_col='id')
b = pd.read_csv(DATA_PATHS['B']['train'], index_col='id')
c = pd.read_csv(DATA_PATHS['C']['train'], index_col='id')

aX, ay = encode_dataset(a)
bX, by = encode_dataset(b)
cX, cy = encode_dataset(c)

ax_train, ax_test, ay_train, ay_test = prepare_data(aX, ay)
bx_train, bx_test, by_train, by_test = prepare_data(bX, by)
cx_train, cx_test, cy_train, cy_test = prepare_data(cX, cy)


a_model, a_loss = train_evaluate(ax_train, ay_train, ax_test, ay_test, str(MODEL_DIR / 'a_dnn_model_replica.tf'))
b_model, b_loss = train_evaluate(bx_train, by_train, bx_test, by_test, str(MODEL_DIR / 'b_dnn_model.tf'))
c_model, c_loss = train_evaluate(cx_train, cy_train, cx_test, cy_test, str(MODEL_DIR / 'c_dnn_model.tf'))

print("Loss Country A: {0:f}".format(a_loss))
print("Loss Country B: {0:f}".format(b_loss))
print("Loss Country C: {0:f}".format(c_loss))
print("Averaged loss: {0:f}".format(
    average_loss(
        [a_loss,  b_loss, c_loss], 
        [len(ax_train), len(bx_train), len(cx_train)]
    ))
)



## Prepare submission:
# load test data
a_test = pd.read_csv(DATA_PATHS['A']['test'], index_col='id')
#a_keep = [e.split('_')[0] for e in ax_train.columns.tolist()]
#a_keep = [elem for n, elem in enumerate(a_keep) if elem not in L[:n]]
a_test = prepare_submission_data(a_test, ax_train.columns.tolist())
a_preds = get_predictions(a_model, a_test)
a_sub = make_country_sub(a_preds, a_test, 'A')

b_test = pd.read_csv(DATA_PATHS['B']['test'], index_col='id')
b_test = prepare_submission_data(b_test, bx_train.columns.tolist())
b_preds = get_predictions(b_model, b_test)
b_sub = make_country_sub(b_preds, b_test, 'B')

c_test = pd.read_csv(DATA_PATHS['C']['test'], index_col='id')
c_test = prepare_submission_data(c_test, cx_train.columns.tolist())
c_preds = get_predictions(c_model, c_test)
c_sub = make_country_sub(c_preds, c_test, 'C')

submission = pd.concat([a_sub, b_sub, c_sub])
submission[submission.poor < 0] = 0
submission.to_csv('submission_dnn.csv')
print("Submission saved.")
