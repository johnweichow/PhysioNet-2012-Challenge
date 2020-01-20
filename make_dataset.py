#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import os
import wget
import zipfile

# save raw data to disk and unzip data
filepath = os.path.join('.', 'data', 'raw')
os.makedirs(filepath)
wget.download('https://physionet.org/files/challenge-2012/1.0.0/set-a.zip', filepath)
wget.download('https://physionet.org/files/challenge-2012/1.0.0/Outcomes-a.txt', filepath)

with zipfile.ZipFile(os.path.join(filepath, 'set-a.zip'), 'r') as zip_ref:
    zip_ref.extractall(filepath)

# join patient files into one dataframe
df = pd.DataFrame()
fp = os.path.join('.', 'data', 'raw', 'set-a')
for filename in os.listdir(fp):
    with open(os.path.join(fp, filename), 'r') as openfile:
        df_temp = pd.read_csv(openfile)
        df_temp['RecordID'] = df_temp[df_temp['Parameter']=='RecordID'].at[0,'Value'].astype(int) # create RecordID column
        df = pd.concat([df, df_temp], ignore_index=True)
# set negative values to null
df.loc[df['Value']<0, 'Value'] = np.nan

# remove outliers
df.loc[(df['Parameter']=='pH') & (df['Value']<6.8), 'Value'] = np.nan
df.loc[(df['Parameter']=='pH') & (df['Value']>8), 'Value'] = np.nan

df.loc[(df['Parameter']=='Temp') & (df['Value']<30), 'Value'] = np.nan
df.loc[(df['Parameter']=='Temp') & (df['Value']>45), 'Value'] = np.nan

df.loc[(df['Parameter']=='HR') & (df['Value']<1), 'Value'] = np.nan
df.loc[(df['Parameter']=='HR') & (df['Value']>250), 'Value'] = np.nan

df.loc[(df['Parameter']=='RespRate') & (df['Value']<1), 'Value'] = np.nan

df.loc[(df['Parameter']=='Weight') & (df['Value']<20), 'Value'] = np.nan


# create a separate dataframe w/ general descriptors
df_gen_desc = df.loc[df['Time']=='00:00', :].copy()
df_gen_desc = df_gen_desc.loc[df['Parameter'].isin(['Weight', 'Gender', 'Height', 'ICUType', 'Age'])]

# Get only the last value per parameter per patient, as some patients have multiple values for Age
df_gen_desc = df_gen_desc.groupby(['RecordID', 'Parameter'])[['Value']].last().reset_index()

# drop general descriptors from original dataframe, except for weight, which is recorded as a timeseries
df = df[~df['Parameter'].isin(['RecordID', 'Gender', 'Height', 'ICUType', 'Age'])]

# convert time from string to minutes elapsed
df['Time'] = df['Time'].map(lambda x: int(x.split(':')[0])*60 + int(x.split(':')[1]))

# save interim data to disk
filepath = os.path.join('.', 'data', 'interim')
os.makedirs(filepath)
os.chdir(filepath)
df.to_csv('set-a_timeseries.csv', index=False)
df_gen_desc.to_csv('set-a_general-descriptors.csv', index=False)
os.chdir(os.path.join(os.pardir, os.pardir))
