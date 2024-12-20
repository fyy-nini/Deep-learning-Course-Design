import numpy as np
from autogluon.tabular import TabularDataset,TabularPredictor
import pandas as pd
train_csv=pd.read_csv('downloads/136244/train.csv')
id,label='id','subscribe'
train_data=TabularDataset(train_csv)
predictor=TabularPredictor(label=label).fit(train_data.drop(columns=[id]))
test_csv=pd.read_csv('downloads/136244/test.csv')
preds=predictor.predict(test_csv.drop(columns=[id]))
submission=pd.DataFrame({id:test_csv[id],label:preds})
submission.to_csv('downloads/136244/submission-autogluon.csv',index=False)


'''No path specified. Models will be saved in: "AutogluonModels/ag-20241218_130825"
Verbosity: 2 (Standard Logging)
=================== System Info ===================
AutoGluon Version:  1.2
Python Version:     3.10.14
Operating System:   Linux
Platform Machine:   x86_64
Platform Version:   #1 SMP Wed Sep 15 17:27:09 CST 2021
CPU Count:          8
Memory Avail:       2.40 GB / 6.00 GB (39.9%)
Disk Space Avail:   362.17 GB / 491.04 GB (73.8%)
===================================================
No presets specified! To achieve strong results with AutoGluon, it is recommended to use the available presets. Defaulting to `'medium'`...
	Recommended Presets (For more details refer to https://auto.gluon.ai/stable/tutorials/tabular/tabular-essentials.html#presets):
	presets='experimental' : New in v1.2: Pre-trained foundation model + parallel fits. The absolute best accuracy without consideration for inference speed. Does not support GPU.
	presets='best'         : Maximize accuracy. Recommended for most users. Use in competitions and benchmarks.
	presets='high'         : Strong accuracy with fast inference speed.
	presets='good'         : Good accuracy with very fast inference speed.
	presets='medium'       : Fast training time, ideal for initial prototyping.
Beginning AutoGluon training ...
AutoGluon will save models to "/mnt/workspace/AutogluonModels/ag-20241218_130825"
Train Data Rows:    22500
Train Data Columns: 20
Label Column:       subscribe
AutoGluon infers your prediction problem is: 'binary' (because only two unique label-values observed).
	2 unique label values:  ['no', 'yes']
	If 'binary' is not the correct problem_type, please manually specify the problem_type parameter during Predictor init (You may specify problem_type as one of: ['binary', 'multiclass', 'regression', 'quantile'])
Problem Type:       binary
Preprocessing data ...
Selected class <--> label mapping:  class 1 = yes, class 0 = no
	Note: For your binary classification, AutoGluon arbitrarily selected which label-value represents positive (yes) vs negative (no) class.
	To explicitly set the positive_class, either rename classes to 1 and 0, or specify positive_class in Predictor init.
Using Feature Generators to preprocess the data ...
Fitting AutoMLPipelineFeatureGenerator...
	Available Memory:                    2458.86 MB
	Train Data (Original)  Memory Usage: 15.25 MB (0.6% of available memory)
	Inferring data type of each feature based on column values. Set feature_metadata_in to manually specify special dtypes of the features.
	Stage 1 Generators:
		Fitting AsTypeFeatureGenerator...
			Note: Converting 1 features to boolean dtype as they only contain 2 unique values.
	Stage 2 Generators:
		Fitting FillNaFeatureGenerator...
	Stage 3 Generators:
		Fitting IdentityFeatureGenerator...
		Fitting CategoryFeatureGenerator...
			Fitting CategoryMemoryMinimizeFeatureGenerator...
	Stage 4 Generators:
		Fitting DropUniqueFeatureGenerator...
	Stage 5 Generators:
		Fitting DropDuplicatesFeatureGenerator...
	Types of features in original data (raw dtype, special dtypes):
		('float', [])  :  5 | ['emp_var_rate', 'cons_price_index', 'cons_conf_index', 'lending_rate3m', 'nr_employed']
		('int', [])    :  5 | ['age', 'duration', 'campaign', 'pdays', 'previous']
		('object', []) : 10 | ['job', 'marital', 'education', 'default', 'housing', ...]
	Types of features in processed data (raw dtype, special dtypes):
		('category', [])  : 9 | ['job', 'marital', 'education', 'default', 'housing', ...]
		('float', [])     : 5 | ['emp_var_rate', 'cons_price_index', 'cons_conf_index', 'lending_rate3m', 'nr_employed']
		('int', [])       : 5 | ['age', 'duration', 'campaign', 'pdays', 'previous']
		('int', ['bool']) : 1 | ['contact']
	0.2s = Fit runtime
	20 features in original data used to generate 20 features in processed data.
	Train Data (Processed) Memory Usage: 1.94 MB (0.1% of available memory)
Data preprocessing and feature engineering runtime = 0.19s ...
AutoGluon will gauge predictive performance using evaluation metric: 'accuracy'
	To change this, specify the eval_metric parameter of Predictor()
Automatically generating train/validation split with holdout_frac=0.1, Train Rows: 20250, Val Rows: 2250
User-specified model hyperparameters to be fit:
{
	'NN_TORCH': [{}],
	'GBM': [{'extra_trees': True, 'ag_args': {'name_suffix': 'XT'}}, {}, {'learning_rate': 0.03, 'num_leaves': 128, 'feature_fraction': 0.9, 'min_data_in_leaf': 3, 'ag_args': {'name_suffix': 'Large', 'priority': 0, 'hyperparameter_tune_kwargs': None}}],
	'CAT': [{}],
	'XGB': [{}],
	'FASTAI': [{}],
	'RF': [{'criterion': 'gini', 'ag_args': {'name_suffix': 'Gini', 'problem_types': ['binary', 'multiclass']}}, {'criterion': 'entropy', 'ag_args': {'name_suffix': 'Entr', 'problem_types': ['binary', 'multiclass']}}, {'criterion': 'squared_error', 'ag_args': {'name_suffix': 'MSE', 'problem_types': ['regression', 'quantile']}}],
	'XT': [{'criterion': 'gini', 'ag_args': {'name_suffix': 'Gini', 'problem_types': ['binary', 'multiclass']}}, {'criterion': 'entropy', 'ag_args': {'name_suffix': 'Entr', 'problem_types': ['binary', 'multiclass']}}, {'criterion': 'squared_error', 'ag_args': {'name_suffix': 'MSE', 'problem_types': ['regression', 'quantile']}}],
	'KNN': [{'weights': 'uniform', 'ag_args': {'name_suffix': 'Unif'}}, {'weights': 'distance', 'ag_args': {'name_suffix': 'Dist'}}],
}
Fitting 13 L1 models, fit_strategy="sequential" ...
Fitting model: KNeighborsUnif ...
	0.8631	 = Validation score   (accuracy)
	0.03s	 = Training   runtime
	0.03s	 = Validation runtime
Fitting model: KNeighborsDist ...
	0.86	 = Validation score   (accuracy)
	0.03s	 = Training   runtime
	0.02s	 = Validation runtime
Fitting model: LightGBMXT ...
	0.8884	 = Validation score   (accuracy)
	1.19s	 = Training   runtime
	0.02s	 = Validation runtime
Fitting model: LightGBM ...
	0.8844	 = Validation score   (accuracy)
	0.62s	 = Training   runtime
	0.01s	 = Validation runtime
Fitting model: RandomForestGini ...
	0.8836	 = Validation score   (accuracy)
	5.89s	 = Training   runtime
	0.08s	 = Validation runtime
Fitting model: RandomForestEntr ...
	0.884	 = Validation score   (accuracy)
	5.76s	 = Training   runtime
	0.07s	 = Validation runtime
Fitting model: CatBoost ...
	0.884	 = Validation score   (accuracy)
	13.71s	 = Training   runtime
	0.01s	 = Validation runtime
Fitting model: ExtraTreesGini ...
	0.8764	 = Validation score   (accuracy)
	2.36s	 = Training   runtime
	0.08s	 = Validation runtime
Fitting model: ExtraTreesEntr ...
	0.8778	 = Validation score   (accuracy)
	2.28s	 = Training   runtime
	0.08s	 = Validation runtime
Fitting model: NeuralNetFastAI ...
No improvement since epoch 9: early stopping
	0.8818	 = Validation score   (accuracy)
	47.13s	 = Training   runtime
	0.08s	 = Validation runtime
Fitting model: XGBoost ...
	0.8862	 = Validation score   (accuracy)
	1.28s	 = Training   runtime
	0.02s	 = Validation runtime
Fitting model: NeuralNetTorch ...
	0.8853	 = Validation score   (accuracy)
	42.3s	 = Training   runtime
	0.08s	 = Validation runtime
Fitting model: LightGBMLarge ...
	0.8876	 = Validation score   (accuracy)
	1.91s	 = Training   runtime
	0.07s	 = Validation runtime
Fitting model: WeightedEnsemble_L2 ...
	Ensemble Weights: {'LightGBMXT': 0.45, 'NeuralNetTorch': 0.45, 'XGBoost': 0.1}
	0.8911	 = Validation score   (accuracy)
	0.15s	 = Training   runtime
	0.0s	 = Validation runtime
AutoGluon training complete, total runtime = 126.19s ... Best model: WeightedEnsemble_L2 | Estimated inference throughput: 19311.0 rows/s (2250 batch size)
Disabling decision threshold calibration for metric `accuracy` due to having fewer than 10000 rows of validation data for calibration, to avoid overfitting (2250 rows).
	`accuracy` is generally not improved through threshold calibration. Force calibration via specifying `calibrate_decision_threshold=True`.
TabularPredictor saved. To load, use: predictor = TabularPredictor.load("/mnt/workspace/AutogluonModels/ag-20241218_130825")'''