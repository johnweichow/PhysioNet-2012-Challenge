#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import os
import math
from tsfresh.feature_extraction import extract_features

df_ICU_onehot = pd.get_dummies(df_gen_desc['ICUType'], prefix='ICUType', drop_first=True)
def build_features(set_name):
	# import data
	filepath = os.path.join('.', 'data')
	df_ts = pd.read_csv(os.path.join(filepath, 'interim', 'set-'+set_name+'_timeseries.csv'))
	df_gen_desc = pd.read_csv(os.path.join(filepath, 'interim', 'set-'+set_name+'_general-descriptors.csv'))
	df_outcomes = pd.read_csv(os.path.join(filepath, 'raw', 'Outcomes-'+set_name+'.txt'))

	# pivot general descriptors so each row is one ID
	df_gen_desc = (
		pd.pivot_table(df_gen_desc, values='Value', index='RecordID', columns='Parameter')
			.reset_index()
	)

	# one-hot encode ICUType
	df_gen_desc.loc[:, 'ICUType'] = df_gen_desc.loc[:, 'ICUType'].apply(int)
	df_ICU_onehot = pd.get_dummies(df_gen_desc['ICUType'], prefix='ICUType', drop_first=True)
	df_gen_desc.drop('ICUType', axis=1, inplace=True)
	df_gen_desc = pd.merge(df_gen_desc, df_ICU_onehot, left_index=True, right_index=True, how='left')

	# create list of features to calculate for each patient parameter. 
	default_fc_params_custom = {
		'median': None,
			'variance': None,
			'maximum': None,
			'minimum': None,
			'length': None, #denotes count
			'linear_trend': [{'attr': 'slope'}]
	}

	# Create features that count the occurences of parameters in certain ranges. 
	# Approach is based on SAPS-II scoring system, where a patient's risk score 
	# increases for values that are too low or high
	kind_to_fc_params_custom = {
			'HR': {'range_count': [{'min': 120, 'max': 9999},
														{'min': 70, 'max': 119},
														{'min': 40, 'max': 69},
														{'min': 0, 'max': 39}]},
			'SysABP': {'range_count': [{'min': 200, 'max': 9999},
														{'min': 100, 'max': 199},
														{'min': 70, 'max': 99},
														{'min': 0, 'max': 69}]},
			'Na': {'range_count': [{'min': 145, 'max': 9999},
														{'min': 125, 'max': 144},
														{'min': 0, 'max': 124}]},
			'K': {'range_count': [{'min': 5.0, 'max': 9999},
														{'min': 3.0, 'max': 4.9},
														{'min': 0, 'max': 3.0}]},
			'WBC': {'range_count': [{'min': 20.1, 'max': 9999},
														{'min': 1.01, 'max': 20},
														{'min': 0, 'max': 1.0}]},
	}

	df_ts_tsfresh = extract_features(df_ts.dropna().set_index('Time'), column_id = 'RecordID',
																		column_kind = 'Parameter', column_value='Value',
																		default_fc_parameters=default_fc_params_custom,
																		kind_to_fc_parameters=kind_to_fc_params_custom)

	df_ts_tsfresh = df_ts_tsfresh.reset_index().rename(columns={'id':'RecordID'})

	# for features that measure the count of a parameter, impute nulls with 0
	for colname in df_ts_tsfresh.columns:
		if '__length' in colname or 'range_count' in colname:
			df_ts_tsfresh[colname].fillna(0, inplace=True)

	# define helper functions to handle dataframes from custom timeseries calculations
	def drop_and_flatten(df):
			df.columns = df.columns.droplevel(0) #drops unwanted 'Value level'
			df.columns = ['_'.join(col).rstrip('_') for col in df.columns.values] #flatten columns
			df.reset_index(inplace=True)
			return df

	# get first and last values in timeseries for each patient parameter
	df_ts_first_last = (
		df_ts.loc[:, ['RecordID', 'Parameter', 'Value']]
					.groupby(['RecordID', 'Parameter'])
					.agg(['first', 'last'])
					.unstack()
	)

	df_ts_first_last = df_ts_first_last.swaplevel(axis=1)
	df_ts_first_last = drop_and_flatten(df_ts_first_last)

	# get mean value of each patient parameter over each 6 hour period
	TIME_PERIOD = 6
	df_ts['Time_group'] = df_ts['Time'].apply(lambda x:
		'hrs-' 
		+ str(math.trunc((x-1)/(TIME_PERIOD * 60))*TIME_PERIOD + 1)
		+ '-'
		+ str((math.trunc((x-1)/(TIME_PERIOD * 60))+1)*TIME_PERIOD)
	)

	df_ts_means = (
		df_ts.loc[:, ['RecordID', 'Parameter', 'Time_group', 'Value']]
			.groupby(['RecordID', 'Parameter', 'Time_group'])
			.agg(['mean'])
			.unstack()
			.unstack()
	)

	df_ts_means = df_ts_means.swaplevel(i=-2, j=-1, axis=1)
	df_ts_means = df_ts_means.swaplevel(i=-3, j=-2, axis=1)
	df_ts_means = drop_and_flatten(df_ts_means)

	# merge timeseries features with general descriptors into one dataframe
	df_features = (
		pd.merge(df_gen_desc, df_ts_tsfresh, how='left',on='RecordID', validate='1:1')
			.merge(df_ts_first_last, how='left', on='RecordID', validate='1:1')
			.merge(df_ts_means, how='left', on='RecordID', validate='1:1')
	)

	# merge features with outcomes
	df_features_outcomes = (
		pd.merge(df_features, df_outcomes.loc[:, ['RecordID', 'In-hospital_death']],
						how='left', on='RecordID', validate='1:1')
	)

	# save processed data to disk
	export_folder = os.path.join('.', 'data', 'processed')
	if not os.path.isdir(export_folder):
		os.makedirs(export_folder)
	filepath = os.path.join(export_folder, 'features_and_outcomes-'+set_name+'.csv')
	df_features_outcomes.to_csv(filepath, index=False)

build_features('a')
build_features('c')