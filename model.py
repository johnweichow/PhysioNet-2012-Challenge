#!/usr/bin/env python
# coding: utf-8

'''
TO DO: 
4. Create Outputs file
'''

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

'''
Score tracker and helper functions go here
'''

class ScoreTracker:
    
    def __init__(self):
        self.score_dict_ = {'model_name':[], 'mean_test_score':[], 'std_test_score':[]}
    
    def add_best_score(self, GridSearchCV, model_name):
        '''
        Adds mean CV score and score standard deviation for a GridSearchCV object
        
        Parameters
        ----------
        GridSearchCV : string
			GridSearchCV object that has been fit to data, to retrieve best score

        model_name : string
			Specifies model name
        '''
        cv_results = pd.DataFrame(GridSearchCV.cv_results_)
        mean_test_score = cv_results.loc[cv_results['rank_test_score']==1, 'mean_test_score'].iloc[0]
        std_test_score = cv_results.loc[cv_results['rank_test_score']==1, 'std_test_score'].iloc[0]
        self.score_dict_['model_name'].append(model_name)
        self.score_dict_['mean_test_score'].append(mean_test_score)
        self.score_dict_['std_test_score'].append(std_test_score)

def score_model(estimator, param_grid, X_train, y_train, ScoreTracker, model_name):
	'''
	Performs a GridSearchCV on estimator using param_grid, X_train, and y_train.
	Stores the best score into ScoreTracker with the specified model_name
	'''
	gscv = GridSearchCV(estimator=estimator, cv=3, param_grid=param_grid,
	                  	verbose=0, n_jobs=-1, scoring='f1',
						return_train_score=False)

	gscv.fit(X_train, y_train)
	ScoreTracker.add_best_score(GridSearchCV=gscv, model_name=model_name)

# create a ScoreTracker to track model performance
scoretracker = ScoreTracker()

# try Logistic Regression
lr = LogisticRegression(solver='liblinear', random_state=42)

lr_baseline_params = {'penalty': ['l1', 'l2']}

score_model(lr, lr_baseline_params, X_train_med_imp_ss, y_train,
	  	    scoretracker, 'Logistic Regression | Median Imputed, Standard Scaling')

# try Random Forest
rf = RandomForestClassifier(random_state=42)

rf_baseline_params = {'n_estimators': [100]}

score_model(rf, rf_baseline_params, X_train_med_imp_ss, y_train,
	  	    scoretracker, 'Random Forest | Median Imputed, Standard Scaling')

score_model(rf, rf_baseline_params, X_train_med_imp, y_train,
	  	    scoretracker, 'Random Forest | Median Imputed')

score_model(rf, rf_baseline_params, X_train_oor_imp, y_train,
	  	    scoretracker, 'Random Forest | Out-of-range Imputed')

# try Balanced Random Forest
brf = BalancedRandomForestClassifier(random_state=42)

brf_baseline_params = {'n_estimators': [100]}

score_model(brf, brf_baseline_params, X_train_med_imp_ss, y_train,
	  	    scoretracker, 'Balanced Random Forest | Median Imputed, Standard Scaling')

score_model(brf, brf_baseline_params, X_train_med_imp, y_train,
	  	    scoretracker, 'Balanced Random Forest | Median Imputed')

score_model(brf, brf_baseline_params, X_train_oor_imp, y_train,
	  	    scoretracker, 'Balanced Random Forest | Out-of-range Imputed')

# Balanced Random Forest on median imputed data is performing the best out of the
# baseline models. Perform RandomizedSearchCV to find a range for each
# hyperparameter for further optimization
brf_random_params = {
	'n_estimators': [10, 100, 1000, 2000],
	'max_depth': [10, 100, None],
	'min_samples_split': [2, 8, 32, 128],
	'min_samples_leaf': [1, 2, 8, 32, 128], 
	'max_features': ['auto', 'log2', None],
	'bootstrap': [True, False]
}

rscv = RandomizedSearchCV(brf, brf_random_params, n_iter=200, 
	                      scoring='f1', verbose=1, n_jobs=-1,
						  cv=3)
rscv.fit(X_train_med_imp, y_train)
print(rscv.best_score_)

# Perform grid search to finalize hyperparameter tuning
brf_grid_params = {
	'n_estimators': [1000, 2000, 4000],
	'max_depth': [2, 10],
	'min_samples_split': [2],
	'min_samples_leaf': [2], 
	'max_features': ['auto'],
	'bootstrap': [True]
}

brf_gscv = GridSearchCV(estimator=brf, cv=3, param_grid=brf_grid_params,
						verbose=1, n_jobs=-1, scoring='f1')
brf_gscv.fit(X_train_med_imp, y_train)
brf_gscv.best_params_