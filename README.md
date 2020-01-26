# Predicting Mortality of Patients in the ICU
-----------
This project creates a machine learning model to predict Intensive Care Unit (ICU) mortality using data from the [PhysioNet 2012 Challenge](https://physionet.org/content/challenge-2012/1.0.0/). 

### Project Overview
Data from 12,000 ICU stays was provided to challenge participants. For each ICU stay, the dataset includes time-series data on up to 37 patient parameters (e.g., albumin levels, heart rate, urine output, etc.) from a patient's first 48 hours in the ICU, as well as up to 6 general descriptors that were only recorded when the patient was admitted (e.g., age, gender, height, etc). 

The competition tasked participants with predicting whether or not each patient survived their hospital stay. During the competition, outcomes were made available for 4,000 of the 12,000 patients (Set A). Participants used this dataset to train their models, and final model scoring was performed using a separate 4,000 patients where outcomes were not made publically available (Set C).

After the competition ended, Set C outcomes were made publically available, so I use this data to check how my modeling approach would have performed had I entered the competition in 2012. 

### Running the Script
Run the python scripts in the following order:
1. `make_dataset.py`: downloads and cleans the data
2. `build_features.py`: engineers features and transforms data to have 1 row per patient

Now, you can now run the notebooks in the repo:
* `exploratory_data_analysis`: exporatory data analysis used to drive feature engineering
* `modeling_approach`: feature selection, model selection, model evaluation, and model interpretation