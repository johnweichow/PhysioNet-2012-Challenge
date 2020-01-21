#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import os

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

from imblearn.ensemble import BalancedRandomForestClassifier

# import data and split out features and outcomes
filepath = os.path.join('.', 'data', 'processed', 'features_and_outcomes.csv')
df = pd.read_csv(filepath)

X = df.drop('In-hospital_death', axis=1)
y = df['In-hospital_death']

# perform train_test_split, ensure it is stratified because the dataset is imablanced
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=42, stratify=y)

# create imputed datasets: median and out-of-range
median_imputer = SimpleImputer(strategy='median')
oor_imputer = SimpleImputer(strategy='constant', fill_value=-999)
X_train_med_imp = median_imputer.fit_transform(X_train)
X_train_oor_imp = oor_imputer.fit_transform(X_train)

# use standard scaling on median imputed data for use with regularized logistic regression
ss = StandardScaler()
X_train_med_imp_ss = ss.fit_transform(X_train_med_imp)

# try Logistic Regression
lr = LogisticRegression(solver='liblinear', random_state=42)

# use a grid search so we can apply stratified k-fold cross validation
lr_baseline_params = {
    'penalty': ['l1', 'l2'],
}

lr_baseline = GridSearchCV(estimator=lr, cv=3, param_grid=lr_baseline_params,
                           verbose=0, n_jobs=-1, scoring='f1',
                           return_train_score=False)

lr_baseline.fit(X_train_med_imp_ss, y_train)

print(pd.DataFrame(lr_baseline.cv_results_).mean_test_score.max().round(
    3), ' LR, median imputation, standardized')


# try Random Forest
rf = RandomForestClassifier(random_state=42)

rf_baseline_params = {
	'n_estimators': [100]
}

rf_baseline = GridSearchCV(estimator=rf, cv=3, param_grid=rf_baseline_params,
                           verbose=0, n_jobs=-1, scoring='f1',
                           return_train_score=False)

rf_baseline.fit(X_train_med_imp, y_train)
print(pd.DataFrame(rf_baseline.cv_results_).mean_test_score.max().round(
    3), ' RF, median imputation')

rf_baseline.fit(X_train_oor_imp, y_train)
print(pd.DataFrame(rf_baseline.cv_results_).mean_test_score.max().round(
    3), ' RF, out-of-range imputation')

# try Balanced Random Forest
brf = BalancedRandomForestClassifier(random_state=42)

brf_baseline = GridSearchCV(estimator=brf, cv=3, param_grid=rf_baseline_params,
                            verbose=0, n_jobs=-1, scoring='f1',
                            return_train_score=False)

brf_baseline.fit(X_train_med_imp, y_train)
print(pd.DataFrame(brf_baseline.cv_results_).mean_test_score.max().round(
    3), ' BRF, median imputation')

brf_baseline.fit(X_train_oor_imp, y_train)
print(pd.DataFrame(brf_baseline.cv_results_).mean_test_score.max().round(
    3), ' BRF, out-of-range imputation')
