import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils._1_Imports.svmI import *
from utils._2_Cleaning.svmC import *
from utils._3_ModelTraining.svmT import *

def metrics_score(actual, predicted):
    """
     Compute and plot confusion matrix. This is a wrapper around : func : ` classification_report ` and
     
     @param actual - The ground truth value.
     @param predicted - The predicted value ( s ) for ` actual `
    """
    print(classification_report(actual, predicted))
    cm = confusion_matrix(actual, predicted)
    plt.figure(figsize=(8,5))
    sns.heatmap(cm, annot=True,  fmt='.2f', xticklabels=['Not Attrite', 'Attrite'], yticklabels=['Not Attrite', 'Attrite'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    
def evaluation_LogisticRegression(lg,x_train,y_train,x_test,y_test):
    """
     Evaluates Logistic Regression on training and testing sets. Prints LR accuracy on training set and test set
     
     @param lg - Classifier to use for evaluation
     @param x_train - List of features to train on.
     @param y_train - List of labels to test on.
    """
    print("LR Accuracy on Train set: ")
    y_pred_train = lg.predict(x_train)
    metrics_score(y_train, y_pred_train)
    
    print("LR Accuracy on Test set: ")
    y_pred_test = lg.predict(x_test)
    metrics_score(y_test, y_pred_test)
    
def evaluation_SupportVectorMachine(svmmodel,x_train,y_train,x_test,y_test):
    """
     Evaluate SVM accuracy on support vector machine. In this case we use Linear Kernel to predict the data
     
     @param svmmodel - SVM model to be evaluated
     @param x_train - list of features for training set ( numpy array )
     @param y_train - list of labels for training set ( numpy array )
     @param x_test - list of features for test set ( numpy array
     @param y_test
    """
    print("SVM Accuracy using Linear Kernel on TRAIN set: ")
    y_pred_train_svm = svmmodel.predict(x_train)
    metrics_score(y_train, y_pred_train_svm)
    
    print("SVM Accuracy using Linear Kernel on TEST set: ")
    y_pred_test_svm = svmmodel.predict(x_test)
    metrics_score(y_test, y_pred_test_svm)
    
def evaluation_SVM_RBF(svmmodelRBF,x_train,y_train,x_test,y_test):
    """
     Evaluates SVM using RBF Kernel on Train and Test sets and prints the accuracy. This is a wrapper for the predict and score methods of the SVM model
     
     @param svmmodelRBF - A scikit - learn SVM model
     @param x_train - A numpy array of shape [ n_samples n_features ]
     @param x_test - A numpy array of shape [ n_samples n_features ]
     @param y_test - A numpy array of shape [ n
    """
    print("SVM Accuracy using RBF Kernel on Train set: ")
    y_pred_train_svm = svmmodelRBF.predict(x_train)
    metrics_score(y_train, y_pred_train_svm)
    
    print("SVM Accuracy using RBF Kernel on Test set: ")
    y_pred_test_svm = svmmodelRBF.predict(x_test)
    metrics_score(y_test, y_pred_test_svm)
    
    
if __name__ == "__main__":
    # Test Code: Specify the path
    file_path = 'Dataset/HR_Employee_Attrition.xlsx'
    # Test Execution: Load the data and check it
    df = load_data(file_path)
    df,num_cols,cat_cols = pre_prep_data(df)
    analyse_numerical_and_categorical_columns(df,num_cols,cat_cols)
    df,X,Y,x_train,x_test,y_train,y_test,X_scaled = cleanprep_and_splitdata(df)
    lg = train_LogisticRegressionModel(x_train,y_train)
    evaluation_LogisticRegression(lg,x_train,y_train)
    svmmodel = train_SupportVectorMachines(x_train,y_train)
    evaluation_SupportVectorMachine(svmmodel,x_train,y_train,x_test,y_test)
    svmmodelRBF = train_SVM_RBF_KERNEL(x_train,y_train)
    evaluation_SVM_RBF(svmmodelRBF,x_train,x_test,y_test)
