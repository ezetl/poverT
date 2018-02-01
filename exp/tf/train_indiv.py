#!/usr/bin/env python3
from pathlib import Path
from tf_utils import *
import pandas as pd
import tensorflow as tf


MODEL_DIR = Path.cwd() / 'models'
LOGS_DIR = Path.cwd() / 'logs'
COUNTRY = sys.argv[1] 

# Keep columns from individual data datasets. Obtained from notebook 8
if COUNTRY == 'A':
    keep_indiv_cols = ['iid', 'gtnNTNam', 'OdXpbPGJ', 'XONDGWjH', 'FPQrjGnS', 'UsmeXdIS', 'igHwZsYz', 'AoLwmlEH', 'QvgxCmCV', 'ukWqmeSS', 'poor', 'country']
elif COUNTRY == 'B':
    keep_indiv_cols = ['iid', 'TJGiunYp', 'esHWAAyG', 'fwRLKHlN', 'dnmwvCng', 'nkxrhykC', 'uDVGRBVU', 'wJthinfa', 'ulQCDoYe', 'poor', 'country']
elif COUNTRY == 'C':
    keep_indiv_cols = ['iid', 'AOSWkWKB', 'XKQWlRjk', 'tOfmAJyI', 'XKyOwsRR', 'CgAkQtOd', 'gxCmCVUk', 'qGqYslGF', 'ShCKQiAy', 'VGJlUgVG', 'poor', 'country']
else:
    raise Exception('Country must be A, B, or C. You provided {}'.format(COUNTRY))


## 1 - PREPARE DATASETS
train_set = pd.read_csv(DATA_PATHS[COUNTRY]['train'], index_col='id')

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
train_reduc.poor.fillna(False, inplace=True)
y_train = np.ravel(train_reduc.poor.astype(int))
del train_reduc['poor']

# CV
test_size = 0.2
x_train, x_test, y_train, y_test = prepare_data(X_train, y_train, test_size=test_size, xgb_format=False)

# Train Neural Net
model, loss = train_evaluate(x_train, y_train, x_test, y_test, epochs=500, model_dir=str(MODEL_DIR / '{}_dnn.tf'.format(COUNTRY)))

print("Loss Country {0}: {1:f}".format(COUNTRY, loss))

## Prepare submission:
# Load and prepare csvs
test_set = pd.read_csv(DATA_PATHS[COUNTRY]['test'], index_col='id')
indexes = test_set.index.get_level_values(0)
test_set_ind = pd.read_csv(DATA_PATHS_IND[COUNTRY]['test'], index_col='id')
keep_test_cols = keep_indiv_cols
keep_test_cols.remove('poor')
test_set_ind = test_set_ind[keep_test_cols]

test_set_ind = prepare_indiv_hhold_set(test_set_ind)
test_set = test_set.merge(test_set_ind, how='left', left_index=True, right_index=True)

# Delete new columns that were not in training set
orig_cols = X_train.columns.tolist() 
keep_cols = train_reduc.columns.tolist()
test_set = test_set[keep_cols]
test_set = pre_process_data(test_set)

diff = set(test_set.columns.tolist()) - set(orig_cols)
test_set = test_set[test_set.columns.difference(list(diff))]
# Add dummy columns that are not in the test set
diff = set(orig_cols) - set(test_set.columns.tolist())
test_set = test_set.assign(**{c: 0 for c in diff})
# Reorder columns in the original order
test_set = test_set[orig_cols]
test_set.fillna(0, inplace=True)

preds = get_predictions(model, test_set)
sub = make_country_sub(preds, test_set, COUNTRY)
grouped_sub = sub.groupby(sub.index.get_level_values(0)).mean()
submission = pd.DataFrame(grouped_sub)
submission['country'] = COUNTRY 
# Reorder columns
submission = submission[['country', 'poor']]
# Reorder indexes
submission = submission.reindex(indexes)

submission.to_csv('submission_dnn_{}.csv'.format(COUNTRY))
print("Submission saved.")
