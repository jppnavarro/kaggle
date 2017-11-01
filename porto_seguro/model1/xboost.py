

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

print

# 0.64167 (+/-0.00527) for {'colsample_bytree': 0.8, 'learning_rate': 0.1, 'min_child_weight': 9, 'n_estimators': 240, 'subsample': 0.8, 'max_depth': 4, 'gamma': 0.2}, gini: 0.28333