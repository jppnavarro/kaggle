
# Based on https://www.kaggle.com/bertcarremans/data-preparation-exploration.

import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from categorical import target_encode


def generate_meta(input_data):
    data = []
    for f in input_data.columns:
        # Defining the role
        if f == 'target':
            role = 'target'
        elif f == 'id':
            role = 'id'
        else:
            role = 'input'

        # Defining the level
        if 'bin' in f or f == 'target':
            level = 'binary'
        elif 'cat' in f or f == 'id':
            level = 'nominal'
        elif input_data[f].dtype == float:
            level = 'interval'
        elif input_data[f].dtype == int:
            level = 'ordinal'

        # Initialize keep to True for all variables except for id
        keep = True
        if f == 'id':
            keep = False

        # Defining the data type
        dtype = input_data[f].dtype

        # Creating a Dict that contains all the metadata for the variable
        f_dict = {
            'varname': f,
            'role': role,
            'level': level,
            'keep': keep,
            'dtype': dtype
        }
        data.append(f_dict)

    meta = pd.DataFrame(data, columns=['varname', 'role', 'level', 'keep', 'dtype'])
    meta.set_index('varname', inplace=True)
    return meta

pd.set_option('display.max_columns', 100)


dir = os.path.dirname(__file__)
train = pd.read_csv(os.path.join(dir, '../input/train.csv'))
test = pd.read_csv(os.path.join(dir, '../input/test.csv'))

#print(train.head())
print("Shape: ", train.shape)

print("\nDroping duplicates...")
num_samples = train.shape[0]
train.drop_duplicates()
if train.shape[0] == num_samples:
    print("No duplicates presented.")
else:
    print("Duplicates droped: %d" % (num_samples - input_data.shape[0]))

print("\nTrain Info:")
print(train.info())

# To facilitate the data management, we'll store meta-information about the variables in a DataFrame.
# This will be helpful when we want to select specific variables for analysis, visualization, modeling, etc.
meta = generate_meta(train)
print()
print(meta)

# Example to extract all nominal variables that are not dropped
print()
print(meta[(meta.level == 'nominal') & (meta.keep)].index)

# Descriptive statistics.
# We can also apply the describe method on the dataframe.
# However, it doesn't make much sense to calculate the mean, std, on categorical
# variables and the id variable. We'll explore the categorical variables visually later.

# Thanks to our meta file we can easily select the variables on
# which we want to compute the descriptive statistics.
# To keep things clear, we'll do this per data type.

v = meta[(meta.level == 'interval') & (meta.keep)].index
print()
print(train[v].describe())

# Overall, we can see that the range of the interval variables is rather small.
# Perhaps some transformation (e.g. log) is already applied in order to anonymize the data?

# Dealing with imbalanced data:
# https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/
def balance(train):
    desired_apriori = 0.10

    # Get the indices per target value
    idx_0 = train[train.target == 0].index
    idx_1 = train[train.target == 1].index

    # Get original number of records per target value
    nb_0 = len(train.loc[idx_0])
    nb_1 = len(train.loc[idx_1])

    # Calculate the undersampling rate and resulting number of records with target=0
    undersampling_rate = ((1 - desired_apriori) * nb_1) / (nb_0 * desired_apriori)
    undersampled_nb_0 = int(undersampling_rate * nb_0)
    print('Rate to undersample records with target=0: {}'.format(undersampling_rate))
    print('Number of records with target = 0 after undersampling: {}'.format(undersampled_nb_0))

    # Randomly select records with target=0 to get at the desired a priori
    undersampled_idx = shuffle(idx_0, random_state=37, n_samples=undersampled_nb_0)

    # Construct list with remaining indices
    idx_list = list(undersampled_idx) + list(idx_1)

    # Return undersample data frame
    return train.loc[idx_list].reset_index(drop=True)

print("\nBalancing training set...")
train = balance(train)

print("\nNumber of training samples target 0: %d" % len(train.loc[train[train.target == 0].index]))
print("Number of training samples target 1: %d" % len(train.loc[train[train.target == 1].index]))

def check_missing_values(train):
    vars_with_missing = []

    for f in train.columns:
        missings = train[train[f] == -1][f].count()
        if missings > 0:
            vars_with_missing.append(f)
            missings_perc = missings / train.shape[0]

            print('Variable {} has {} records ({:.2%}) with missing values'.format(f, missings, missings_perc))

    print('In total, there are {} variables with missing values'.format(len(vars_with_missing)))

check_missing_values(train)

# Dropping the variables with too many missing values
def drop_missing_variables(train):
    vars_to_drop = ['ps_car_03_cat', 'ps_car_05_cat']
    train.drop(vars_to_drop, inplace=True, axis=1)
    meta.loc[(vars_to_drop), 'keep'] = False  # Updating the meta

    # Imputing with the mean or mode
    mean_imp = Imputer(missing_values=-1, strategy='mean', axis=0)
    mode_imp = Imputer(missing_values=-1, strategy='most_frequent', axis=0)
    train['ps_reg_03'] = mean_imp.fit_transform(train[['ps_reg_03']]).ravel()
    train['ps_car_12'] = mean_imp.fit_transform(train[['ps_car_12']]).ravel()
    train['ps_car_14'] = mean_imp.fit_transform(train[['ps_car_14']]).ravel()
    train['ps_car_11'] = mode_imp.fit_transform(train[['ps_car_11']]).ravel()

drop_missing_variables(train)

# Checking the cardinality of the categorical variables.
def check_cardinality(train):
    v = meta[(meta.level == 'nominal') & (meta.keep)].index

    for f in v:
        dist_values = train[f].value_counts().shape[0]
        print('Variable {} has {} distinct values'.format(f, dist_values))


# Cardinality refers to the number of different values in a variable.
# As we will create dummy variables from the categorical variables later on,
# we need to check whether there are variables with many distinct values.
check_cardinality(train)

train_encoded, test_encoded = target_encode(train["ps_car_11_cat"],
                                            test["ps_car_11_cat"],
                                            target=train.target,
                                            min_samples_leaf=100,
                                            smoothing=10,
                                            noise_level=0.01)

train['ps_car_11_cat_te'] = train_encoded
train.drop('ps_car_11_cat', axis=1, inplace=True)
meta.loc['ps_car_11_cat', 'keep'] = False  # Updating the meta
test['ps_car_11_cat_te'] = test_encoded
test.drop('ps_car_11_cat', axis=1, inplace=True)


# Creating dummy variables.
# The values of the categorical variables do not represent any
# order or magnitude. For instance, category 2 is not twice the value of category 1.
# Therefore we can create dummy variables to deal with that.
# We drop the first dummy variable as this information can
# be derived from the other dummy variables generated for the categories of the original variable.
v = meta[(meta.level == 'nominal') & (meta.keep)].index
print('\nBefore dummification we have {} variables in train'.format(train.shape[1]))
train = pd.get_dummies(train, columns=v, drop_first=True)
print('After dummification we have {} variables in train'.format(train.shape[1]))


# Creating interaction variables
v = meta[(meta.level == 'interval') & (meta.keep)].index
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
interactions = pd.DataFrame(data=poly.fit_transform(train[v]), columns=poly.get_feature_names(v))
# Remove the original columns.
interactions.drop(v, axis=1, inplace=True)
# Concat the interaction variables to the train data.
print('\nBefore creating interactions we have {} variables in train'.format(train.shape[1]))
train = pd.concat([train, interactions], axis=1)
print('After creating interactions we have {} variables in train'.format(train.shape[1]))


# Removing features with low or zero variance
# Personally, I prefer to let the classifier algorithm
# chose which features to keep. But there is one thing
# that we can do ourselves. That is removing features with
# no or a very low variance. Sklearn has a handy method to do that:
# VarianceThreshold. By default it removes features with zero variance.
# This will not be applicable for this competition as we saw there are
# no zero-variance variables in the previous steps. But if we would remove
# features with less than 1% variance, we would remove 31 variables.
selector = VarianceThreshold(threshold=.01)
selector.fit(train.drop(['id', 'target'], axis=1)) # Fit to train without id and target variables

f = np.vectorize(lambda x : not x) # Function to toggle boolean array elements

v = train.drop(['id', 'target'], axis=1).columns[f(selector.get_support())]
print('\n{} variables have too low variance.'.format(len(v)))
print('These variables are {}'.format(list(v)))

# Selecting features with a Random Forest and SelectFromModel
# Here we'll base feature selection on the feature importances of a random forest.
# With Sklearn's SelectFromModel you can then specify how many variables you want to keep.
# You can set a threshold on the level of feature importance manually. But we'll simply select the top 50% best variables.

# The code in the cell below is borrowed from the GitHub repo of Sebastian Raschka.
# https://github.com/rasbt/python-machine-learning-book/blob/master/code/ch04/ch04.ipynb
# This repo contains code samples of his book Python Machine Learning, which is an absolute must to read.

X_train = train.drop(['id', 'target'], axis=1)
y_train = train['target']

feat_labels = X_train.columns
print('Training random forest...')
rf = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)
rf.fit(X_train, y_train)
importances = rf.feature_importances_
indices = np.argsort(rf.feature_importances_)[::-1]

print()
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,feat_labels[indices[f]], importances[indices[f]]))


# With SelectFromModel we can specify which prefit classifier to use and
# what the threshold is for the feature importances.
# With the get_support method we can then limit the number of variables in the train data.
sfm = SelectFromModel(rf, threshold='median', prefit=True)
print('\nNumber of features before selection: {}'.format(X_train.shape[1]))
n_features = sfm.transform(X_train).shape[1]
print('Number of features after selection: {}'.format(n_features))
selected_vars = list(feat_labels[sfm.get_support()])

train = train[selected_vars + ['target']]

# Feature scaling
# As mentioned before, we can apply standard scaling to the training data.
# Some classifiers perform better when this is done.
scaler = StandardScaler()
scaler.fit_transform(train.drop(['target'], axis=1))


#------------------------------------------
"""
X = train.drop(['target'], axis=1)
y = train['target'].values

from sklearn.model_selection import GridSearchCV
from xgboost.sklearn import XGBClassifier

params = {'colsample_bytree': [0.8],
          'learning_rate': [0.1],
          'min_child_weight': [9],
          'n_estimators': [240],
          'subsample': [0.8],
          'max_depth': [4],
          'gamma': [0.2]}

clf = GridSearchCV(estimator = XGBClassifier(objective= 'binary:logistic', scale_pos_weight=1,seed=27),
 param_grid = params, scoring='roc_auc',n_jobs=-1, iid=False, cv=5, verbose=5)
clf.fit(X,y)

print("Melhores parametros:")
print
print(clf.best_params_)
print
print("Scores do grid search nos dados de treinamento (todos):")
print
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']

for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.5f (+/-%0.05f) for %r, gini: %0.5f" % (mean, std * 2, params, 2*mean - 1))
"""