#!/usr/bin/env python
# coding: utf-8

'''
TO DO: 
Create Outputs file
'''

import pandas as pd
import numpy as np
import os

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, make_scorer
from sklearn.decomposition import PCA

from imblearn.ensemble import BalancedRandomForestClassifier

# import data and split out features and outcomes
filepath = os.path.join('.', 'data', 'processed', 'features_and_outcomes.csv')
df = pd.read_csv(filepath)

X = df.drop(['RecordID', 'In-hospital_death'], axis=1)
y = df['In-hospital_death']

# perform train_test_split, ensure it is stratified because the dataset is imablanced
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=42, stratify=y)

# create imputed datasets: median and out-of-range
median_imputer = SimpleImputer(strategy='median')
oor_imputer = SimpleImputer(strategy='constant', fill_value=-999)
X_train_med_imp = median_imputer.fit_transform(X_train)
X_test_med_imp = median_imputer.transform(X_test)

X_train_oor_imp = oor_imputer.fit_transform(X_train)
X_test_oor_imp = oor_imputer.transform(X_test)

# use standard scaling on median imputed data for use with regularized logistic regression
ss = StandardScaler()
X_train_med_imp_ss = ss.fit_transform(X_train_med_imp)
X_test_med_imp_ss = ss.transform(X_test_med_imp)

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

print(scoretracker.score_dict_)

# Balanced Random Forest on median imputed data is performing the best out of the
# baseline models. Perform RandomizedSearchCV to find a range for each
# hyperparameter for further optimization
brf_random_params = {
	'n_estimators': [10, 100, 200, 1000],
	'max_depth': [4, 10, None],
	'min_samples_split': [2, 8, 32, 128],
	'min_samples_leaf': [1, 2, 8, 32, 128], 
	'max_features': ['auto', 'log2', None],
	'bootstrap': [True]
}

rscv = RandomizedSearchCV(brf, brf_random_params, n_iter=200, 
	                      scoring='f1', verbose=1, n_jobs=-1,
						  cv=3)
rscv.fit(X_train_med_imp, y_train)
print(rscv.best_score_)
print(rscv.best_params_)

# Perform grid search to finalize hyperparameter tuning
brf_grid_params = {
	'n_estimators': [25, 50, 100, 200],
	'max_depth': [2, 4, 8],
	'min_samples_split': [4, 8, 16, 32],
	'min_samples_leaf': [1, 2, 4], 
	'max_features': ['auto', 'log2'],
	'bootstrap': [True]
}

brf_gscv = GridSearchCV(estimator=brf, cv=3, param_grid=brf_grid_params,
						verbose=1, n_jobs=-1, scoring='f1')
brf_gscv.fit(X_train_med_imp, y_train)

best_brf = brf_gscv.best_estimator_
best_brf.fit(X_train_med_imp, y_train)

def get_best_threshold(fit_model, X_test, y_test):
    '''
    Given a fit model, returns classification threshold that maximizes contest score for X_test, y_test
    '''
    predicted_probabilities = fit_model.predict_proba(X_test)[:,1] # get prediction probabilities for death
    
    best_score = -1
    best_threshold = -1
    best_precision = -1
    best_recall = -1
    best_f1 = -1

    for threshold in np.linspace(0,1,101):
        preds_at_threshold = (predicted_probabilities >= threshold).astype(int)
        if sum(preds_at_threshold) == 0: 
            f1 = 0
            precision = 0
            recall=0
        else:
            f1 = f1_score(y_test, preds_at_threshold)
            precision = precision_score(y_test, preds_at_threshold)
            recall = recall_score(y_test, preds_at_threshold)
            contest_score = min(precision, recall) # use the scoring system of the actual competiton
            if contest_score > best_score:
                best_score = contest_score
                best_threshold = threshold
                best_precision = precision
                best_recall = recall
                best_f1 = f1
    print(type(fit_model).__name__)
    print('Contest Score of {:2.2%} at threshold of {:2.0%}'.format(best_score, best_threshold))
    print('(Precision: {:2.2%} | Recall: {:2.2%} | F1: {:2.2%}))'.format(best_precision, best_recall, best_f1))
    return best_threshold

def get_score_at_threshold(fit_model, X_test, y_test, threshold):
    '''
    Given a fit model and classification threshold, returns contest score by predicting probabilities for
    X_test, y_test at the given threshold
    '''
    predicted_probabilities = fit_model.predict_proba(X_test)[:,1] # get prediction probabilities for death
    preds_at_threshold = (predicted_probabilities >= threshold).astype(int)
    precision = precision_score(y_test, preds_at_threshold)
    recall = recall_score(y_test, preds_at_threshold)
    f1 = f1_score(y_test, preds_at_threshold)
    score = min(precision, recall)
    print(type(fit_model).__name__)
    print('Expected Contest Score of {:2.2%} at threshold of {:2.0%}'.format(score, threshold))
    print('(Precision: {:2.2%} | Recall: {:2.2%} | F1: {:2.2%}))'.format(precision, recall, f1))
    print('\n')

# get threshold that maximizes contest score for training data
best_threshold = get_best_threshold(best_brf, X_train_med_imp, y_train)
get_score_at_threshold(best_brf, X_test_med_imp, y_test, best_threshold)

rf.fit(X_train_med_imp, y_train)
best_threshold = get_best_threshold(rf, X_train_med_imp, y_train)
get_score_at_threshold(rf, X_test_med_imp, y_test, best_threshold)

lr_grid_params = {
    'penalty': ['l2'],
    'C': [10**x for x in np.linspace(-4, 1, 10)]
}

lr_gcsv = GridSearchCV(estimator=lr, param_grid=lr_grid_params, cv = 3, verbose = 2,
                       scoring='f1', n_jobs=-1, return_train_score=True)
lr_gcsv.fit(X_train_med_imp_ss, y_train)

best_lr = lr_gcsv.best_estimator_

best_lr.fit(X_train_med_imp_ss, y_train)

# get threshold that maximizes contest score for training data
best_threshold = get_best_threshold(best_lr, X_train_med_imp_ss, y_train)

# check expected contest score on test data, using that threshold 
get_score_at_threshold(best_lr, X_test_med_imp_ss, y_test, best_threshold)


pca = PCA(n_components=100)
X_train_pca = pca.fit_transform(X_train_med_imp_ss)
X_test_pca = pca.transform(X_test_med_imp_ss)

best_brf = brf_gscv.best_estimator_
best_brf.fit(X_train_pca, y_train)
best_threshold = get_best_threshold(best_brf, X_train_pca, y_train)
get_score_at_threshold(best_brf, X_test_pca, y_test, best_threshold)