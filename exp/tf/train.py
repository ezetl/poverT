#!/usr/bin/env python3
import tensorflow as tf
import pandas as pd
from sklearn.utils import resample
import sys
from pathlib import Path
sys.path.append(str(Path.cwd() / '..' / 'notebooks'))
# export PYTHONPATH=~/Software/xgboost/python-package<Paste>
from utils import *


MODEL_DIR = Path.cwd() / 'models'
LOGS_DIR = Path.cwd() / 'logs'


# load training data
a = pd.read_csv(DATA_PATHS['A']['train'], index_col='id')
b = pd.read_csv(DATA_PATHS['B']['train'], index_col='id')
c = pd.read_csv(DATA_PATHS['C']['train'], index_col='id')

aX, ay = encode_dataset(a)
bX, by = encode_dataset(b)
cX, cy = encode_dataset(c)

# Prepare data to train
ax_train, ax_test, ay_train, ay_test = prepare_data(aX, ay)
bx_train, bx_test, by_train, by_test = prepare_data(bX, by)
cx_train, cx_test, cy_train, cy_test = prepare_data(cX, cy)


def get_feature_columns(df):
    feat_cols = [] 
    for col in df: 
        feat_cols.append(tf.feature_column.numeric_column(col))
    return feat_cols


def get_input_fn(x, y, features=None, num_epochs=800, shuffle=True):
  return tf.estimator.inputs.pandas_input_fn(
      x=pd.DataFrame({k: x[k].values for k in features}),
      y=pd.Series(y),
      num_epochs=num_epochs,
      shuffle=shuffle)



model_a = tf.estimator.LinearClassifier(
    model_dir=MODEL_DIR, feature_columns=get_feature_columns(ax_train))
model_b = tf.estimator.LinearClassifier(
    model_dir=MODEL_DIR, feature_columns=get_feature_columns(bx_train))
model_c = tf.estimator.LinearClassifier(
    model_dir=MODEL_DIR, feature_columns=get_feature_columns(cx_train))



a_feat_cols = get_feature_columns(ax_train)
a_regressor = tf.estimator.DNNRegressor(
    feature_columns=a_feat_cols,
    hidden_units=[100, 500, 50],
    model_dir=str(MODEL_DIR / 'a_dnn_model.tf')
)


a_regressor.train(input_fn=get_input_fn(ax_train, ay_train, features=ax_train.columns.tolist()), steps=5000)


ev = a_regressor.evaluate(
    input_fn=get_input_fn(ax_test, ay_test, features=ax_test.columns.tolist(), num_epochs=1, shuffle=False))

print("Loss Country A: {0:f}".format(ev["loss"]))



