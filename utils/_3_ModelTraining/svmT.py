import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils._1_Imports.svmI import *
from utils._2_Cleaning.svmC import *


def train_LogisticRegressionModel(x_train,y_train):
    """
     Trains Logistic Regression model on training data. This is a wrapper to the fit function
     
     @param x_train - numpy array of shape [ n_samples n_features ]
     @param y_train - numpy array of shape [ n_samples ]
     
     @return a Logistic Regression model trained on training
    """
    lg=LogisticRegression()
    lg.fit(x_train,y_train)
    print("Logistic Regression successfully trained")
    return lg
    
def train_SupportVectorMachines(x_train,y_train):
    """
     Train support vector machine on data. This is a wrapper for SVC.
     
     @param x_train - pandas DataFrame with feature vectors
     @param y_train - pandas Series with labels for training data
     
     @return A model that can be used to predict the features
    """
    svm = SVC(kernel = 'linear') #linear kernal or linear decision boundary
    svmmodel = svm.fit(X = x_train, y = y_train)
    print("Support Vector machine successfully trained")
    return svmmodel

def train_SVM_RBF_KERNEL(x_train,y_train):
    """
     Train SVM with RBF Kernel. This is a wrapper for the SVM's fit method
     
     @param x_train - pandas DataFrame with columns of training data
     @param y_train - pandas Series with columns of training labels
     
     @return - trained SVM with RBF Kernel as a
    """
    svm = SVC(kernel = 'rbf') #RBF
    svmmodelRBF = svm.fit(X = x_train, y = y_train)
    print("SVM with RBF Kernel successfully trained")

    return svmmodelRBF

if __name__ == "__main__":
    # Test Code: Specify the path
    file_path = 'Dataset/HR_Employee_Attrition.xlsx'
    # Test Execution: Load the data and check it
    df = load_data(file_path)
    df,num_cols,cat_cols = pre_prep_data(df)
    analyse_numerical_and_categorical_columns(df,num_cols,cat_cols)
    df,X,Y,x_train,x_test,y_train,y_test,X_scaled = cleanprep_and_splitdata(df)
    train_LogisticRegressionModel(x_train,y_train)
    train_SupportVectorMachines(x_train,y_train)
    train_SVM_RBF_KERNEL(x_train,y_train)
