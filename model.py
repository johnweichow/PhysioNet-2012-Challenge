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

filepath = os.path.join('.', 'data', 'processed', 'features_and_outcomes.csv')
df = pd.read_csv(filepath)