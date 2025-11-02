
'''
#main_functions.py

Functions to apply UMAP dimensionality reduction, trial supervised learning classification, and plot 
continuous or categorical variables in embedding plots (static, interactive w/ plotly). 
Trials are saved in separate folders.

#Followed documentation: 
https://umap-learn.readthedocs.io/en/latest/api.html
https://umap-learn.readthedocs.io/en/latest/plotting.html

https://plotly.com/python/hover-text-and-formatting/
https://stackoverflow.com/questions/14947909/python-checking-to-which-bin-a-value-belongs
https://seaborn.pydata.org/generated/seaborn.color_palette.html

#Documentation on Supervised learning
https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
https://knowledge.dea.ga.gov.au/notebooks/Real_world_examples/Scalable_machine_learning/3_Evaluate_optimize_fit_classifier/

Created: 20-Oct-25, Marco Acevedo
Updated: 29-Oct-25, Marco; 


'''

import os
import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import Normalize

import joblib #for saving UMAP model
import seaborn as sns
import colorcet as cc
import plotly.express as px 
sns.set_theme(style='white', context='notebook', rc={'figure.figsize':(14, 10)}) #anything smaller does not help with points

#Advanced plotting libraries
from itertools import compress
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from bokeh.plotting import output_notebook #for interactive plot

#Dimensionality reduction
import umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#Machine learning
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, ShuffleSplit, KFold
from sklearn.metrics import roc_curve, auc, balanced_accuracy_score, f1_score

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.inspection import DecisionBoundaryDisplay
from skimage import measure

#Geochemistry libraries
import pyrolite.geochem
import pyrolite.comp
import subprocess #for calling R from terminal


#region Helper functions

def make_dir(destDir):
	image_dir = destDir
	if not os.path.exists(image_dir):
		os.makedirs(image_dir)  

def find_outliers_iqr(df, column):
	#Interquartile Range (IQR) Method for skewed or non-normally distributed data.

	Q1 = df[column].quantile(0.25)
	Q3 = df[column].quantile(0.75)
	IQR = Q3 - Q1
	lower_bound = Q1 - 1.5 * IQR
	upper_bound = Q3 + 1.5 * IQR
	idx_outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
	return idx_outliers

#endregion

#region Load table functions

#Developed for categorical and continuous variables, for example:
#ml_classes, setID, Dataset	District,	Deposit_Batholith,	Temporality,	Timing,	Position,	Composition,	Continent
#'P', 'Ce', 'Eu', 'Th', 'Hf', 'Imputed_La', 'Imputed_Pr', 'Imputed_Y', 'Imputed_Nd', 'Imputed_Gd', 'Imputed_Er', 'Imputed_Yb', 'Imputed_Sm', 'Imputed_Dy', 'Imputed_Lu'

def load_dataset_type0(filepath1, end_idx_categorical, end_idx_numerical):
	#custom this function according to new tables (update type):
	#convention followed: all categoricals must be on the left-hand side of the table

	# list_change = ["Temporality", "Deposit_Batholith", "Position", "Continent", 
	# 			"Composition", "District", "ID", "Sample", "Suite"] #"Zircon.Name"
	
	#default
	# end_idx_categorical = 19 #last raw input categorical column
	# end_idx_numerical = 64 #last raw input numerical column

	#Load table (the column indexes below can be obtained from a data dictionary)
	table1 = pd.read_csv(filepath1, low_memory=False)

	#Generating workable table
	range_data = list(range(0, end_idx_numerical)) #for merged dataset from 'merge_Carrasco_db_v2.m'
	table2 = table1.iloc[:, range_data] 
	column_names = table2.columns	

	#medicine 1: avoid slash / in column names
	condition1 = 'Deposit/Batholith' in column_names #in Carrasco tables
	condition2 = 'DepositBatholith' in column_names	#in Elise tables	
	if condition1:
		table2 = table2.rename(columns={'Deposit/Batholith': 'Deposit_Batholith'}) 
	elif condition2:
		table2 = table2.rename(columns={'DepositBatholith': 'Deposit_Batholith'}) 

	column_names2 = table2.columns
	categorical_columns = column_names2[0:end_idx_categorical]
	numerical_columns = column_names2[end_idx_categorical:end_idx_numerical]

	#medicine 2: replace NA with Unknown	
	table2[categorical_columns] = table2[categorical_columns].astype(str)
	for item in categorical_columns:
		idx_temp = table2[item].isnull()
		table2.loc[idx_temp, item] = 'Unknown'

	#medicine 3: convert to double
	# table2[numerical_columns] = table2[numerical_columns].astype('float64')
	
	# table2[numerical_columns] = pd.to_numeric(table2[numerical_columns], errors='coerce')
	for col in numerical_columns:
		table2[col] = pd.to_numeric(table2[col], errors='coerce') #non-numeric strings to NaN		

	# #medicine 4: replace NaN with zero (old Elise script)
	# list_change2 = ["U", "Th", "Ti", "Hf", "Nb", "Ta", "Ce"]
	# for item in list_change2:
	# 	idx_temp = table2[item].isnull()
	# 	table2.loc[idx_temp, item] = 0
	
	#Save
	filepath2 = os.path.join(os.path.dirname(filepath1), 'input_table.csv')
	table2.to_csv(filepath2, index=False) #processed table	

	return table2

def load_dataset_type1(filepath1, filepath3, filepath3_umap, requested_variables, data_start_idx):
	#custom this function according to new tables (update type):
	
	#Load table (the column indexes below can be obtained from a data dictionary)
	table1 = pd.read_csv(filepath1, low_memory=False)

	#Generating workable table
	range_imputed = list(requested_variables) #range(65, 77)
	table2 = table1.iloc[:, range_imputed] 

	#medicine 1: avoid / in column names
	try:
		table2 = table2.rename(columns={'Deposit/Batholith': 'Deposit_Batholith'}) #Carrasco tables
	except:
		table2 = table2.rename(columns={'DepositBatholith': 'Deposit_Batholith'}) #Elise tables

	#medicine 2: replace NA with Unknown
	list_change = ["Temporality", "Deposit_Batholith", "Position", "Continent", "Composition", "Timing", "District"]
	for item in list_change:
		idx_temp = table2[item].isnull()
		table2.loc[idx_temp, item] = 'Unknown'

	#medicine 3: dropping rows with empty cells
	any_idx = table2.isna().any(axis=1)
	table3 = table2.loc[np.invert(any_idx), :] 
	table3.reset_index(inplace = True) #the index from the input table is preserved (for searching points)

	#medicine 4: dropping categorical variables (feeback required)	
	input_umap = table3.iloc[:, data_start_idx:] 

	col_names = list(input_umap.columns)
	col_names2 = list(table3.iloc[:, 0:data_start_idx].columns)

	print(f"Table 2 has {table2.shape[0]} rows and Table 3 (without NA rows) has {table3.shape[0]}")
	print(f"Categoricals: {col_names2}")
	print(f"Continuous (for UMAP): {col_names}")

	table3.to_excel(filepath3, index=False) #processed table
	input_umap.to_excel(filepath3_umap, index=False)

	return table3, input_umap



#endregion

#region Geochemical calculations

def geochemistry_calculation1(table_input):

	Nd_array = table_input[["Imputed_Nd"]].to_numpy()
	Ce_array = table_input[["Ce"]].to_numpy()
	Dy_array = table_input[["Imputed_Dy"]].to_numpy()
	Yb_array = table_input[["Imputed_Yb"]].to_numpy()
	Gd_array = table_input[["Imputed_Gd"]].to_numpy()

	ratio1 = Dy_array/Yb_array
	ratio2 = Ce_array/Nd_array
	ratio3 = Gd_array/Yb_array

	array = np.concatenate((ratio1, ratio2, ratio3), axis=1)	
	new_columns = ["Dy_Yb", "Ce_Nd", "Gd_Yb"]
	table_output = pd.DataFrame(array, columns= new_columns)

	return table_output

def impute_REE(data_table, r_script_path, destDir):
	#Imputation of REE. Following Carrasco-Godoy et al. (2023)
	#Criteria for calculation: 
		#'Ce', 'Eu' excluded
		#HREE included (Lu, Yb, Tm)
		#Sm recommended due to uncertainty
	
	x_labels_carrasco = [
		'La', 'Pr', 'Nd', 'Gd', 'Tb', 'Ho', 'Er',
		'Tm', 'Yb', 'Y', 'Sm', 'Lu', 'Dy', 
		'Ce', 'Eu',
		]

	#Input
	df = data_table.loc[:, x_labels_carrasco]	

	#Save
	file1 = 'input_Carrasco.csv'
	filepath1 = os.path.join(destDir, file1)
	df.to_csv(filepath1, sep=",", index=False)

	#Output
	arg1 = destDir
	command = ["Rscript", r_script_path, arg1]

	try:
		# Execute the command and capture output
		# universal_newlines=True ensures output is treated as text
		result = subprocess.run(command, capture_output=True, text=True, check=True)

		# Print the standard output and standard error from the R script
		print("R Script Output:")
		print(result.stdout)
		print("R Script Error (if any):")
		print(result.stderr)

	except subprocess.CalledProcessError as e:
		print(f"Error running R script: {e}")
		print(f"R Script Output: {e.stdout}")
		print(f"R Script Error: {e.stderr}")
	except FileNotFoundError:
		print("Error: Rscript command not found. Ensure R is installed and in your PATH.")

	#Retrieve
	file2 = 'output_Carrasco.csv'	
	filepath2 = os.path.join(destDir, file2) #patternCoeff_output_Chondrite Lattice_.95	
	output_Carrasco = pd.read_table(filepath2, delimiter=',')

	#medicine: avoid duplicate columns later
	output_Carrasco2 = output_Carrasco.drop(columns= x_labels_carrasco, axis=1)
	
	return output_Carrasco2

def anenburg_lambdas(data_table, destDir):
	#Calculate REE pattern coefficients. Following  Anenburg and Williams (2022)

	raw_input = 'no'	
	n_rows = data_table.shape[0]
	id_number = list(range(1, n_rows + 1))

	x_labels_anenburg = [	
		'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd',
		'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 
		'Lu'
		] #for raw data
	
	if raw_input == 'yes':
		
		df = data_table.loc[:, x_labels_anenburg]	

	elif raw_input == 'no':

		x_labels_anenburg1 = [
			'ppmCalc_La', 'Ce', 'ppmCalc_Pr', 'Imputed_Nd', 'Imputed_Sm', 'Eu', 'Imputed_Gd',
			'Imputed_Tb', 'Imputed_Dy', 'Imputed_Ho', 'Imputed_Er', 'Imputed_Tm', 'Imputed_Yb', 
			'Imputed_Lu'
			] #imputed with script_Carrasco.R that excluded Ce and Eu
		
		df = data_table.loc[:, x_labels_anenburg1]	
		df.columns = x_labels_anenburg #renaming
	
	#Input		
	df.insert(0, "id_number", id_number) #for reference

	#Output
	ls = df.pyrochem.lambda_lnREE(degree=4, algorithm="ONeill", 
								exclude=["Ce", "Eu"], anomalies=["Ce", "Eu"], 
								add_X2=True, sigmas=0.1, add_uncertainties=True)

	ls.insert(0, "id_number", id_number)
	
	ls = ls.rename(columns={'λ0': 'lambda_0', 'λ1': 'lambda_1', 'λ2': 'lambda_2', 'λ3': 'lambda_3',
							'λ0_σ': 'lambda_0_sigma', 'λ1_σ': 'lambda_1_sigma', 'λ2_σ': 'lambda_2_sigma', 'λ3_σ': 'lambda_3_sigma', 
							'X2': 'fit_chi_squared', 'Ce/Ce*': 'Ce_ratio_A', 'Eu/Eu*': 'Eu_ratio_A'})

	#Save
	file1 = 'input_Anenburg.csv'
	file2 = 'output_Anenburg.csv'
	filepath1 = os.path.join(destDir, file1)
	filepath2 = os.path.join(destDir, file2)
	df.to_csv(filepath1, sep=",", index=False)
	ls.to_csv(filepath2, sep=",", index=False)

	#Retrieve
	output_Anenburg = pd.read_table(filepath2, delimiter=',')
	
	return output_Anenburg

def centered_log_ratio_transform(X): 	
	"""Centered log-ratio transformation for compositional data."""
		
	#Prevent 'inf' outputs (although, it creates artefacts)
	epsilon_val = 10**-5
	idx_zero = (X == 0)	
	values = X[idx_zero]
	X[idx_zero] = values + epsilon_val

	#Manual implementation of CLR
	if np.any(X <= 0):
		raise ValueError("CLR transformation requires strictly positive values.")
	
	geometric_mean = np.exp(np.mean(np.log(X), axis=1)).reshape(-1, 1)
	clr_X = np.log(X / geometric_mean)

	return clr_X

def aitchison_CLR(table_input, data_folder1):	
	#default
	original_index = table_input.index
	#Note: dedicated to zircon		
	
	table_input['ZrSiO'] = 1_000_000 - table_input.sum(axis=1) 

	column_names = table_input.columns
	output_cols = [col + '_clr' for col in column_names]	

	#Pre-processing
	any_zero_rows = (table_input == 0).any(axis=1)
	table_input_cleaned = table_input[~any_zero_rows]	
	new_index = table_input_cleaned.index

	#CLR (Elise)
	X = table_input_cleaned.to_numpy()
	clr_X = centered_log_ratio_transform(X)
	lr_df = pd.DataFrame(clr_X, columns= output_cols, index= new_index)

	# #Alternative: Aitchison CLR (might produce 'inf')	
	# lr_df = df_cleaned.pyrocomp.CLR()	
	# lr_df.index = new_index
	# lr_df.columns = output_cols

	#Post-processing
	lr_df2 = lr_df.reindex(original_index)

	#Save
	file1 = 'input_CLR.csv'
	file2 = 'output_CLR.csv'
	filepath1 = os.path.join(data_folder1, file1)
	filepath2 = os.path.join(data_folder1, file2) 
	
	table_input.to_csv(filepath1, sep=",", index=False)
	lr_df2.to_csv(filepath2, sep=",", index=False)

	return lr_df2



#endregion

#region PCA

def pca_calculation(table_input, standardise_choice, data_folder1):
	#default	
	n_components = 3	
	new_col_names = [f'PC{i+1}' for i in range(n_components)]
	original_index = table_input.index

	#Pre-processing
	table_input_cleaned0 = table_input.dropna(ignore_index=False)
	new_index = table_input_cleaned0.index	

	if standardise_choice == 'yes':
		scaler = StandardScaler()
		table_input_cleaned = scaler.fit_transform(table_input_cleaned0)

	elif standardise_choice =='no':
		table_input_cleaned = table_input_cleaned0

	#PCA factorisation
	pca = PCA(n_components=n_components)
	principal_components = pca.fit_transform(table_input_cleaned)	

	pca_scores = pd.DataFrame(principal_components, columns=new_col_names,
							  index=new_index)

	loadings = pd.DataFrame(pca.components_.T[:, :3], columns=new_col_names,
							index=table_input_cleaned0.columns)
	
	explained_variance_ratio = pca.explained_variance_ratio_

	#Post-processing
	pca_scores2 = pca_scores.reindex(original_index)

	#Save
	file1 = 'output_PCA.csv'	
	filepath1 = os.path.join(data_folder1, file1)		
	pca_scores2.to_csv(filepath1, sep=",", index=False)		

	return pca_scores2, loadings, explained_variance_ratio

#endregion

#region UMAP

def umap_calculation(table_input_umap, umap_variables, variable_legend, 
					 neighbors_input, min_dist_input, metric_input, components_output, 
					 filepath4, filepath5, data_folder1):
	#default	
	n_components = components_output	
	new_col_names = [f'umap{i+1}' for i in range(n_components)]
	original_index = table_input_umap.index

	input_umap = table_input_umap[umap_variables] #workable table

	#Pre-processing 
	input_umap_cleaned = input_umap.dropna(ignore_index=False)
	new_index = input_umap_cleaned.index
	table_input_umap2 = table_input_umap.loc[new_index]

	#Y data (only for supervised UMAP)
	condition1 = (variable_legend == "")
	if condition1:
		classif_sup = 0		
	else:
		classif_var = table_input_umap2[variable_legend]
		list_unique = classif_var.unique()    
		mapping = {item:i for i, item in enumerate(list_unique)}
		classif_sup = classif_var.apply(lambda x: mapping[x]) #categorical array (same size as table3)				

	input_umap2 = input_umap_cleaned.values #np
	
	embedding = custom_umap_transform(input_umap2, classif_sup, condition1,
								   neighbors_input, min_dist_input, metric_input, components_output,
								   filepath4, filepath5)
	
	#coordinates	
	embedding2 = pd.DataFrame(embedding, columns=new_col_names,
							  index=new_index)
	
	#Post-processing
	output_UMAP = embedding2.reindex(original_index)

	#Save
	file1 = 'input_UMAP.csv'	
	file2 = 'output_UMAP.csv'	
	filepath1 = os.path.join(data_folder1, file1)			
	filepath2 = os.path.join(data_folder1, file2)		
	input_umap.to_csv(filepath1, sep=",", index=False)	
	output_UMAP.to_csv(filepath2, sep=",", index=False)	

	return output_UMAP

def custom_umap_transform(input_umap, classif_sup, condition1, 
						  neighbors_input, min_dist_input, metric_input, components_output, 
						  filepath4, filepath5):
	
	#Default
	seed_chosen = 42 #consistency in transform operations (avoid stochastic variation)	
	verbose_option = False #progress bar
	# dict_tqdm = {'ncols': 1000, 'colour': 'green'}	#tqdm_kwds 

	#X data
	sc = StandardScaler() #sc.mean_ , sc.scale_	
	scaled_data = sc.fit_transform(input_umap) 	

	#object for umap.plot
	if condition1: #non-parametric unsupervised

		umap_model = umap.UMAP(n_neighbors= neighbors_input,
						min_dist= min_dist_input,
					  	metric= metric_input, 
					  	n_components= components_output,
						transform_seed= seed_chosen,
						verbose=verbose_option, 						
						).fit(scaled_data)  		
		
	else: #non-parametric supervised					
		
		umap_model = umap.UMAP(n_neighbors= neighbors_input,
						min_dist= min_dist_input,
						metric= metric_input, 
						n_components= components_output,
						transform_seed= seed_chosen,
						verbose=verbose_option,      						
						).fit(scaled_data, y=classif_sup)
	
	#Note: In future work, please, implement neural network for parametric UMAP

	embedding = umap_model.embedding_ 

	#Saving data for reproducibility	
	joblib.dump(sc, filepath4) #scaler
	joblib.dump(umap_model, filepath5) #umap transform   

	return embedding


def load_umap(filepath4, filepath5, data_folder1):
	#for reproducibility

	#Load tables
	input_umap = pd.read_csv(os.path.join(data_folder1, 'input_UMAP.csv'))	

	sc = joblib.load(filepath4) #scaler
	umap_model = joblib.load(filepath5) #umap transformation
	# embedding = umap_model.embedding_  	

	return input_umap, sc, umap_model

def umap_apply(table_input_umap, umap_variables, sc, umap_model, data_folder1):
	
	#default	
	n_components = umap_model.embedding_.shape[1]
	new_col_names = [f'umap{i+1}' for i in range(n_components)]
	original_index = table_input_umap.index

	input_umap = table_input_umap[umap_variables] #workable table

	#Pre-processing 
	input_umap_cleaned = input_umap.dropna(ignore_index=False)
	new_index = input_umap_cleaned.index	

	#Transformations
	data = input_umap_cleaned.values
	scaled_data = sc.transform(data) 
	embedding = umap_model.transform(scaled_data)
	
	#coordinates	
	embedding2 = pd.DataFrame(embedding, columns=new_col_names,
							  index=new_index)
	
	#Post-processing
	output_UMAP = embedding2.reindex(original_index)

	#Save
	file1 = 'input_UMAP.csv'	
	file2 = 'output_UMAP.csv'	
	filepath1 = os.path.join(data_folder1, file1)			
	filepath2 = os.path.join(data_folder1, file2)		
	input_umap.to_csv(filepath1, sep=",", index=False)	
	output_UMAP.to_csv(filepath2, sep=",", index=False)	

	return output_UMAP

#endregion

#region Machine Learning

def train_binary_classifier(table3, embedding2, variable_legend_ML, type_ml_model, data_folder2):
	
	#default
	# components_output = embedding2.shape[1]
	list_unique_ML = table3[variable_legend_ML].unique()	
	seed = 42 #for reproducibility

	#mapping
	mapping_ML = {item:i for i, item in enumerate(list_unique_ML)}
	classif_ML = table3[variable_legend_ML].apply(lambda x: mapping_ML[x]) #categorical array (same size as table3)

	#Split training and testing sets	
	X = embedding2
	Y = classif_ML.to_numpy()
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state= seed)	

	#Choose model

	#very fast 
	if type_ml_model == 'rt':
		clf = DecisionTreeClassifier(max_depth=5, random_state= seed)

	elif type_ml_model == 'rf':
		clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, random_state= seed)

	elif type_ml_model == 'knn':
		clf = KNeighborsClassifier(n_neighbors=5) 

	elif type_ml_model == 'ada_boost':
		clf = AdaBoostClassifier(n_estimators=80,  learning_rate=1.4, algorithm="SAMME", random_state= seed)

	#fast
	
	#SVC is slower when probability= True
	elif type_ml_model == 'svc_linear':	#support vector machine
		clf = SVC(kernel="linear", C=0.025, random_state= seed, probability= True) #C = 1 (regularisation)

	elif type_ml_model == 'svc_poly':	
		clf = SVC(kernel="poly", degree=2, C=0.25, random_state= seed, probability= True) #degree = 5 (takes ~1 min)

	elif type_ml_model == 'svc_rbf': #radial basis function	
		clf = SVC(kernel= 'rbf', gamma=2, C=1, random_state= seed, probability= True) 

	elif type_ml_model == 'neural_network':	
		clf = MLPClassifier(alpha=1, max_iter=1000, random_state= seed) 

	else:
		print('Wrong type_ml_model input.')

	clf.fit(X_train, y_train) #.best_estimator_

	vert_faces, mesh_2d, mesh_arrays = predict_meshgrid(X, clf)

	# Predict
	#Note: we skipped Nested K-Fold Cross-Validation (future work)
	best_model = clf 
	pred = best_model.predict(X_test)
	ac = balanced_accuracy_score(y_test, pred)
	f1_ = f1_score(y_test, pred) 

	# Receiver operating characteristic (ROC) area under the curve (AUC)
	probs = best_model.predict_proba(X_test)
	probs = probs[:, 1]
	fpr, tpr, thresholds = roc_curve(y_test, probs)
	auc_ = auc(fpr, tpr)

	print(f"Model: {type_ml_model}")
	print("Mean balanced accuracy: "+ str(round(np.mean(ac), 2)))
	print("Mean F1: "+ str(round(np.mean(f1_), 2)))
	print("Mean roc_auc: "+ str(round(np.mean(auc_), 3)))

	#Save ML model and training data for reproducibility	
	filepath1 = os.path.join(data_folder2, 'wiggle_classifier.sav')
	filepath2 = os.path.join(data_folder2, 'wiggle_classifier_X.sav')
	joblib.dump(clf, filepath1) #sklearn model
	joblib.dump(X, filepath2) #training data

	return fpr, tpr, vert_faces

def predict_meshgrid(X, clf):
	#used for 3D surface (binary classification)
	components_output = X.shape[1]
	n_samples = 50
	padding = 0.5

	# Create a 2D meshgrid
	x_min, x_max = X[:, 0].min() - padding, X[:, 0].max() + padding
	y_min, y_max = X[:, 1].min() - padding, X[:, 1].max() + padding		
	

	xx_2d, yy_2d = np.meshgrid(np.linspace(x_min, x_max, n_samples),
								np.linspace(y_min, y_max, n_samples),
								)
	mesh_2d = [xx_2d, yy_2d]

	if components_output == 2:						
		
		grid_points = np.c_[xx_2d.ravel(), yy_2d.ravel()]
		Z_prediction = clf.predict(grid_points).reshape(xx_2d.shape)

		mesh_arrays = [xx_2d, yy_2d, Z_prediction]
		verts_faces = 0

	elif components_output == 3: # Create a 3D meshgrid		
		z_min, z_max = X[:, 2].min() - padding, X[:, 2].max() + padding		
		
		xx_3d, yy_3d, zz_3d = np.meshgrid(np.linspace(x_min, x_max, n_samples),
								np.linspace(y_min, y_max, n_samples),
								np.linspace(z_min, z_max, n_samples))
		
		# Predict on the meshgrid
		grid_points = np.c_[xx_3d.ravel(), yy_3d.ravel(), zz_3d.ravel()]
		Z_prediction = clf.predict(grid_points).reshape(xx_3d.shape)
		# Z_prediction = clf.decision_function(grid_points).reshape(xx.shape)

		mesh_arrays = [xx_3d, yy_3d, zz_3d, Z_prediction]	

		# This approach evaluates decision function across slices in z dimension
		z_range = np.linspace(z_min, z_max, n_samples)
		decision_values = np.empty((n_samples, n_samples, n_samples))

		for i, z_val in enumerate(z_range):

			grid = np.c_[xx_2d.ravel(), yy_2d.ravel(), np.full(xx_2d.size, z_val)]
			try:
				decision_values[:, :, i] = clf.decision_function(grid).reshape(xx_2d.shape)
			except AttributeError:
				decision_values[:, :, i] = clf.predict_proba(grid)[:, 1].reshape(xx_2d.shape) - 0.5

		# Extract the surface closest to decision boundary (≈ 0)
		verts, faces, _, _ = measure.marching_cubes(decision_values, level=0)

		# Convert voxel indices to real coordinates
		verts_coord = np.column_stack([
			np.interp(verts[:, 0], [0, n_samples-1], [x_min, x_max]),
			np.interp(verts[:, 1], [0, n_samples-1], [y_min, y_max]),
			np.interp(verts[:, 2], [0, n_samples-1], [z_min, z_max])
		])

		verts_faces = verts_coord[faces]

	return verts_faces, mesh_2d, mesh_arrays

def load_wiggle(data_folder2):
	filepath1 = os.path.join(data_folder2, 'wiggle_classifier.sav')
	filepath2 = os.path.join(data_folder2, 'wiggle_classifier_X.sav')

	clf = joblib.load(filepath1) 
	X = joblib.load(filepath2) 

	return X, clf

def plot_ROC(fpr, tpr):

	auc_ = auc(fpr, tpr)

	plt.title('Receiver Operating Characteristic')
	plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % auc_)
	plt.legend(loc = 'lower right')
	plt.plot([0, 1], [0, 1],'r--')
	plt.xlim([0, 1])
	plt.ylim([0, 1])
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	plt.show()

#endregion

#region Plot colormaps

def define_categorical_cmap_custom(table3, variable_legend):
	#default
	chosen_palette = cc.glasbey_category10	

	list_unique = table3[variable_legend].unique()
	n_classes = len(list_unique)

	#colourmap
	mapping = {item:i for i, item in enumerate(list_unique)}
	classif= table3[variable_legend].apply(lambda x: mapping[x]) #categorical array (same size as table3)
	colourmap = sns.color_palette(palette= chosen_palette, n_colors = n_classes)
	# colours = [sns.color_palette(palette= colourmap)[x] for x in classif] #RGB triplets

	colourmap_updated = colourmap #pre-allocating
	for x in range(0, n_classes):        
		
		name = list_unique[x]

		colours_sub = colourmap[x]
		
		#Custom replacements (based on modified Carrasco table)
		if name == 'Unknown':
			colours_sub = colors.to_rgb('lightgrey')
		
		if name == 'nan':
			colours_sub = colors.to_rgb('lightgrey')

		if name == 'Ore syn-mineral magmatism':
			colours_sub = colors.to_rgb('red')

		if name == 'Syn Mineral':
			colours_sub = colors.to_rgb('red')

		if name == 'Ore related magmatism':
			target_colour = (255, 208, 0)
			colours_sub = tuple(ti/255 for ti in target_colour)		

		if name == 'S Type Granite':
			colours_sub = colors.to_rgb('violet')

		colourmap_updated[x] = colours_sub

	#converting triplets into string
	colours = [sns.color_palette(palette= colourmap_updated)[x] for x in classif] #RGB triplets
	colours0 = np.array(colours)*255
	colours1 = colours0.round(0)
	a = colours1.astype(str)
	colours2 = [f'rgb({",".join(c)})' for c in a]	

	return list_unique, classif, colourmap_updated, colours2

def define_continuous_cmap(table3, variable_legend, percentile, n_bins, chosen_palette): 

	array = table3[variable_legend]

	#Generating categories  
	values = np.nanpercentile(array, [percentile, 100 - percentile]) #NaN vulnerable
	histogram_range = values[1] - values[0]
	
	intervals = list(np.arange(values[0], values[1], histogram_range/n_bins))
	intervals.append(values[1])
	categorisation = np.digitize(array, intervals)
	df_categorisation = pd.Series(categorisation)

	list_unique = np.unique(categorisation)
	n_classes = len(list_unique) 
	#0= below first interval
	#n_classes= above last interval

	#colourmap
	mapping = {item:i for i, item in enumerate(list_unique)}
	classif= df_categorisation.apply(lambda x: mapping[x]) #categorical array (same size as table3)
	colourmap = sns.color_palette(palette= chosen_palette, n_colors = n_classes) 

	colourmap_updated = colourmap
	
	#converting triplets into string
	colours = [sns.color_palette(palette= colourmap_updated)[x] for x in classif] #RGB triplets
	colours0 = np.array(colours)*255
	colours1 = colours0.round(0)
	a = colours1.astype(str)
	classif_colours = [f'rgb({",".join(c)})' for c in a]	

	return list_unique, classif, colourmap_updated, classif_colours
	
def scale_to_unit_range(arr):
	"""Scale numpy array values to [0,1] range."""
	min_val = np.min(arr)
	max_val = np.max(arr)
	return (arr - min_val) / (max_val - min_val)

def pca_to_rgb(pca_scores):
	r = scale_to_unit_range(pca_scores['PC1'])
	g = scale_to_unit_range(pca_scores['PC2'])
	b = scale_to_unit_range(pca_scores['PC3'])
	return np.vstack([r, g, b]).T  # shape (n_samples, 3)

#endregion

#region PCA plots

def plot_pca_biplots_rgb(pca_scores, rgb_colors):
	# Plot PC1 vs PC2, colored by the RGB colors
	plt.figure(figsize=(8, 6))
	plt.scatter(pca_scores['PC1'], pca_scores['PC2'], c=rgb_colors, s=20,edgecolor='k', linewidth=0.3)  
	plt.xlabel('PC1')
	plt.ylabel('PC2')
	plt.title('PC1 vs PC2 Scatter Plot Colored by PCA RGB')
	plt.grid(True)
	plt.tight_layout()
	plt.show()

	# Plot PC1 vs PC3, colored by the RGB colors
	plt.figure(figsize=(8, 6))
	plt.scatter(pca_scores['PC1'], pca_scores['PC3'], c=rgb_colors, s=20, edgecolor='k', linewidth=0.3)  
	plt.xlabel('PC1')
	plt.ylabel('PC3')
	plt.title('PC1 vs PC3 Scatter Plot Colored by PCA RGB')
	plt.grid(True)
	plt.tight_layout()
	plt.show()

def plot_pca_3d_rgb(pca_scores, rgb_colors):
	
	pcs = pca_scores[['PC1', 'PC2', 'PC3']]

	# Plot 3D scatter plot with RGB colors
	fig = plt.figure(figsize=(9, 7))
	ax = fig.add_subplot(111, projection='3d')

	ax.scatter(pcs['PC1'], pcs['PC2'], pcs['PC3'], c=rgb_colors, s=10, alpha=0.8, edgecolor='k', linewidth=0.3)

	ax.set_xlabel('PC1')
	ax.set_ylabel('PC2')
	ax.set_zlabel('PC3')
	ax.set_title('PCA 3D-scatter Plot Colored by PCA RGB')

	plt.tight_layout()
	plt.show()

def pca_plots(pca_scores, explained_variance_ratio, groups):
	#default
	n_components = 3
	group_column = "ml_classes"

	pca_scores[group_column] = groups

	# Scree plot
	plt.figure(figsize=(8, 5))
	plt.plot(range(1, n_components + 1),
			 explained_variance_ratio, marker='o', linestyle='--')
	plt.title("Scree Plot")
	plt.xlabel("Principal Component")
	plt.ylabel("Explained Variance Ratio")
	plt.grid(True)
	plt.tight_layout()
	plt.show()

	# 2D Scatter Plot: PC1 vs PC2
	plt.figure(figsize=(7, 6))
	sns.scatterplot(data=pca_scores, x='PC1', y='PC2', hue=group_column, palette='Set2')
	plt.title('PC1 vs PC2 Scatter Plot')
	plt.xlabel(f"PC1 ({explained_variance_ratio[0]*100:.1f}%)")
	plt.ylabel(f"PC2 ({explained_variance_ratio[1]*100:.1f}%)")
	plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
	plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
	plt.legend()
	plt.grid(True)
	plt.tight_layout()
	plt.show()

	# 3D Scatter Plot
	fig = plt.figure(figsize=(8, 6))
	ax = fig.add_subplot(111, projection='3d')
	for group in pca_scores[group_column].unique():
		group_data = pca_scores[pca_scores[group_column] == group]
		ax.scatter(group_data['PC1'], group_data['PC2'], group_data['PC3'], label=group)
	ax.set_xlabel(f"PC1 ({explained_variance_ratio[0]*100:.1f}%)")
	ax.set_ylabel(f"PC2 ({explained_variance_ratio[1]*100:.1f}%)")
	ax.set_zlabel(f"PC3 ({explained_variance_ratio[2]*100:.1f}%)")
	ax.set_title("PC1 vs PC2 vs PC3 (3D Scatter Plot)")
	ax.legend()
	plt.tight_layout()
	plt.show()

# Correlation circle plot
def plot_correlation_circle(loadings, title="Correlation Circle"):
	plt.figure(figsize=(7, 7))
	plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
	plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
	circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--')
	plt.gca().add_artist(circle)

	x_axis, y_axis = loadings.columns[0], loadings.columns[1]
	for i in range(loadings.shape[0]):
		plt.arrow(0, 0, loadings.iloc[i, 0], loadings.iloc[i, 1],
				  color='r', alpha=0.7, head_width=0.02)
		plt.text(loadings.iloc[i, 0]*1.1, loadings.iloc[i, 1]*1.1,
				 loadings.index[i], ha='center', va='center', fontsize=9)

	plt.title(title)
	plt.xlabel(x_axis)
	plt.ylabel(y_axis)
	plt.xlim(-1.1, 1.1)
	plt.ylim(-1.1, 1.1)
	plt.gca().set_aspect('equal', adjustable='box')
	plt.grid(True)
	plt.tight_layout()
	plt.show()

#endregion

#region UMAP plots

def plot_umap_biplot_rgb(data, col_x, col_y, rgb_colors):
	plt.figure(figsize=(8, 6))
	plt.scatter(data[col_x], data[col_y], color=rgb_colors, alpha=0.8, edgecolor='k', linewidth=0.3, s=10)
	plt.xlabel(col_x)
	plt.ylabel(col_y)
	plt.title(f'UMAP: {col_x} vs {col_y} colored by PCA RGB')
	plt.grid(True)
	plt.tight_layout()
	plt.show()

def export_legend(legend, filepath2, expand=[-5,-5,5,5]):   

	fig  = legend.figure
	fig.canvas.draw()
	bbox  = legend.get_window_extent()
	bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
	bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
	
	fig.savefig(filepath2, dpi="figure", bbox_inches=bbox)

def plot_umap_variable(embedding2, colourmap_updated, list_unique, classif, 
					   title_str, variable_legend1, markerSize, data_folder2):
	
	#default
	components_output = embedding2.shape[1]
	n_classes = len(list_unique)	

	#Saving names
	filepath4_new = os.path.join(data_folder2, variable_legend1 + "_plot.png")
	filepath3_new = os.path.join(data_folder2, variable_legend1 + "_legend.png")
	
	#Plot
	# markerSize = 30
	fontSize = 18

	if components_output == 2:	
		fig = plt.figure(figsize=(10, 10)) #dpi= 200, figsize=(10, 10)
		ax = plt.gca()
	if components_output == 3:	
		fig = plt.figure(figsize=(10, 10))
		ax = fig.add_subplot(111, projection='3d') 

	for x in range(0, n_classes):
		
		idx = (classif == x)
		name = list_unique[x]
		
		colours_sub = np.asarray(colourmap_updated[x]).reshape(1,-1)        
		
		if components_output == 2:
			scatter = plt.scatter(embedding2[idx, 0], embedding2[idx, 1],
								c=colours_sub, label = name,
								s= markerSize, alpha= .7, edgecolors= 'none')
			
		elif components_output == 3:
			scatter = ax.scatter(embedding2[idx, 0], embedding2[idx, 1], embedding2[idx, 2],
								c=colours_sub, label = name,
								s= markerSize, alpha= .5, edgecolors= 'none')
			
			ax.set_zlabel('z')

	#Settings
	ax.autoscale(False)
	ax.grid(True)
	ax.set_aspect('equal', 'datalim')	
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_title(title_str, fontsize=24)
	
	#Legend
	lgnd = plt.legend(ncol=1, fontsize= fontSize, loc='center right', bbox_to_anchor=(1.4, 0.5),
					markerscale= 10, scatterpoints=1)
	export_legend(lgnd, filepath2= filepath3_new)
	lgnd.remove()	

	plt.show()
	fig.savefig(filepath4_new, dpi="figure")


def plot_umap_variable_wiggle(embedding2, colourmap_updated, list_unique, classif, X, clf, vert_faces,
							  title_str, type_background, variable_legend1, markerSize, data_folder2):
	
	#default
	components_output = embedding2.shape[1]
	n_classes = len(list_unique)		

	#Saving names
	filepath4_new = os.path.join(data_folder2, variable_legend1 + "_wiggle_plot.png")
	filepath3_new = os.path.join(data_folder2, variable_legend1 + "_wiggle_legend.png")

	#Plot
	# markerSize = 20 #default= 4
	fontSize = 18

	if components_output == 2:	
		fig = plt.figure(figsize=(10, 10)) #dpi= 200, figsize=(10, 10)
		ax = plt.gca()
	if components_output == 3:	
		fig = plt.figure(figsize=(10, 10))
		ax = fig.add_subplot(111, projection='3d') 

	for x in range(0, n_classes):
		
		idx = (classif == x)
		name = list_unique[x]
			
		colours_sub = np.asarray(colourmap_updated[x]).reshape(1,-1)        
		
		if components_output == 2:			
			scatter = plt.scatter(x=embedding2[idx, 0], y=embedding2[idx, 1],
								c=colours_sub, label = name,
								s= markerSize, alpha= .7, edgecolors= 'none')
			
			#Wiggles produced by ML classification probabilities			
			cm = plt.cm.RdBu	

			if type_background == 'gradient': #as continuous gradient (image)
				DecisionBoundaryDisplay.from_estimator(clf, X, plot_method='pcolormesh', alpha=0.1, ax=ax) 		

			elif type_background == 'solid': #area with colours following predicted classes (if ignoring 'cm')
				DecisionBoundaryDisplay.from_estimator(clf, X, cmap=cm, alpha=0.05, eps=.5
												, ax=ax, response_method='predict') 		

			elif type_background == 'contour': #include decision boundary and margins (for SVC)
				DecisionBoundaryDisplay.from_estimator(clf, X, cmap=cm, plot_method='contourf',  alpha=0.1, 
												ax=ax, response_method="decision_function") 			

		elif components_output == 3:				
				
			mesh = Poly3DCollection(vert_faces, alpha=0.25, facecolor='grey', edgecolor='none', linewidths=0.1)
			mesh.set_facecolor((.6, 0.6, 0.6, 0.25))  # RGBA for transparency
			ax.add_collection3d(mesh)

			scatter = ax.scatter(embedding2[idx, 0], embedding2[idx, 1], embedding2[idx, 2],
								c=colours_sub, label = name,
								s= markerSize, alpha= 1, edgecolors= 'none',
								zorder=10)				
			
			ax.set_zlabel('z')
			

	#Settings
	ax.autoscale(False)
	ax.grid(True)
	ax.set_aspect('equal', 'datalim')	
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_title(title_str, fontsize=24)

	#Legend
	lgnd = plt.legend(ncol=1, fontsize= fontSize, loc='center right', bbox_to_anchor=(1.4, 0.5),
					markerscale= 10, scatterpoints=1)
	export_legend(lgnd, filepath2= filepath3_new)
	lgnd.remove()				

	plt.show()
	fig.savefig(filepath4_new, dpi="figure")


def plot_umap_variable_interactive(df3, cmap_updated_str, variable_interactive):	
	#2D
	#note: glitch in hover mini-table. Upper plot half is inactive when too many values in the dict_names

	#default
	box_size = 1050 #plot size		
			
	col_names = list(df3.columns)	
	dict_names = collections.OrderedDict.fromkeys(col_names, True) #issue: resorted
	dict_names["classif"] = False		

	fig = px.scatter( df3, x='umap1', y='umap2', color='classif', 
					hover_name= variable_interactive, 
					hover_data= dict_names,					
					width=box_size, height=box_size) #, color_discrete_sequence= colourmap

	fig.update_yaxes(scaleanchor = "x", scaleratio = 1) #range=[None, None]
	fig.update_traces(marker = dict(size= 4, color= cmap_updated_str)) #marker_color
	fig.update_layout(legend=dict(title= None), 
					hoverlabel=dict(bgcolor='rgba(255,255,255,0.65)',
									font=dict(color='black'))
									) #hovermode='x unified', 

	fig.show()
			

#endregion

