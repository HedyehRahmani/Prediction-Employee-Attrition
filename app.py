from utils._1_Imports.svmI import *
from utils._2_Cleaning.svmC import *
from utils._3_ModelTraining.svmT import *
from utils._4_ModelEvaluation.svmE import *

file_path = 'Dataset/HR_Employee_Attrition.xlsx'
# Load data and run training and testing functions on it. This is the main file that will be called by the client
df = load_data(file_path)
df,num_cols,cat_cols = pre_prep_data(df)
analyse_numerical_and_categorical_columns(df,num_cols,cat_cols)
df,X,Y,x_train,x_test,y_train,y_test,X_scaled = cleanprep_and_splitdata(df)
lg = train_LogisticRegressionModel(x_train,y_train)
evaluation_LogisticRegression(lg,x_train,y_train,x_test,y_test)
svmmodel = train_SupportVectorMachines(x_train,y_train)
evaluation_SupportVectorMachine(svmmodel,x_train,y_train,x_test,y_test)
svmmodelRBF = train_SVM_RBF_KERNEL(x_train,y_train)
evaluation_SVM_RBF(svmmodelRBF,x_train,y_train,x_test,y_test)
