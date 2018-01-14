import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
import sys
from pathlib import Path
sys.path.append(str(Path.cwd() / '..' / 'notebooks'))
from utils import *


def get_feature_columns(df):
    feat_cols = [] 
    for col in df: 
        feat_cols.append(tf.feature_column.numeric_column(col))
    return feat_cols


def get_input_fn(x, y=None, features=None, num_epochs=500, shuffle=True):
    if y is not None:
        y = pd.Series(y)
    return tf.estimator.inputs.pandas_input_fn(
      x=pd.DataFrame({k: x[k].values for k in features}),
      y=y,
      num_epochs=num_epochs,
      shuffle=shuffle)


def get_model(feat_cols, model_dir):
    return tf.estimator.DNNRegressor(
        feature_columns=feat_cols,
        hidden_units=[100, 100, 50],
        model_dir=model_dir
    )


def train_evaluate(X, y, X_test, y_test, model_dir):
    feat_cols = get_feature_columns(X)
    model = get_model(feat_cols, model_dir)
    model.train(input_fn=get_input_fn(X, y, features=X.columns.tolist()))
    ev = model.evaluate(
        input_fn=get_input_fn(X_test, y_test, features=X_test.columns.tolist(), num_epochs=1, shuffle=False))
    return model, ev["loss"]


def average_loss(losses, lenghts):
    """Returns the weighted averaged loss"""
    total_len = sum(lenghts)
    return np.average(losses, weights=[lenghts[0] / total_len, lenghts[1] / total_len, lenghts[2] / total_len])


def prepare_submission_data(df_test, keep_columns):
    test = pre_process_data(df_test)
    diff = set(test.columns.tolist()) - set(keep_columns)
    test = test[test.columns.difference(list(diff))]
    diff = set(keep_columns) - set(test.columns.tolist())
    test = test.assign(**{c: 0 for c in diff})
    test = test[keep_columns]
    test.fillna(0, inplace=True)
    return test


def get_predictions(model, test_set):
    preds = model.predict(
        input_fn=get_input_fn(test_set, features=test_set.columns.tolist(), num_epochs=1, shuffle=False))
    # .predict() returns an iterator of dicts; convert to a list and print
    # predictions
    preds = [e['predictions'][0] for e in preds]
    return preds
