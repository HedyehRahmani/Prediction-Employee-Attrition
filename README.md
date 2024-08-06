
# Employee Attrition Prediction Using SVM
`PLEASE REVIEW THE NOTEBOOK FOR STEP-BY-STEP ANAYLSIS`
In today's competitive environment, retaining top talent is crucial for any organization. This repository focuses on predicting employee attrition using Support Vector Machines (SVM), a powerful classification technique. By analyzing various factors, we aim to identify employees who are at risk of leaving the company, enabling proactive retention strategies.

## Table of Contents

- [Context](#context)
- [Project Objective](#project-objective)
- [Dataset Overview](#dataset-overview)
- [Methodology](#methodology)
  - [Data Preprocessing](#data-preprocessing)
  - [Feature Selection](#feature-selection)
  - [Model Training and Tuning](#model-training-and-tuning)
  - [Evaluation Metrics](#evaluation-metrics)
- [Installation Guide](#installation-guide)
- [Running the Project](#running-the-project)
- [Results and Insights](#results-and-insights)
- [Contribution Guidelines](#contribution-guidelines)

## Context

An MNC with a global workforce faces significant costs associated with employee turnover. To address this challenge, the company's Head of People Operations has initiated a project to predict employee attrition. By identifying at-risk employees, the company can target retention efforts more effectively and reduce overall attrition rates.

## Project Objective

The goal of this project is twofold:

1. **Identify Factors Driving Attrition**: By analyzing demographic and job-related factors, we seek to uncover patterns that indicate an employee's likelihood of leaving the company.
2. **Predict Attrition**: Using SVM, we aim to develop a model that accurately predicts whether an employee will attrite, enabling targeted retention strategies.

## Dataset Overview

The dataset contains a wealth of information, including demographic details, work-related metrics, and an attrition flag. Some of the key variables include:

- **EmployeeNumber**: Unique identifier for each employee
- **Attrition**: Binary indicator of whether the employee left the company
- **Age, Gender, MaritalStatus**: Demographic factors
- **JobSatisfaction, EnvironmentSatisfaction, WorkLifeBalance**: Employee satisfaction metrics
- **MonthlyIncome, JobRole, Department**: Job-related variables
- **YearsAtCompany, YearsInCurrentRole**: Tenure and experience indicators

## Methodology

### Data Preprocessing

- **Handling Missing Values**: Imputation techniques are employed to handle any missing data.
- **Feature Scaling**: Z-score normalization is used to scale features, ensuring that all variables contribute equally to the model.

### Feature Selection

- **Correlation Analysis**: Identify highly correlated features to reduce multicollinearity.
- **Domain Expertise**: Incorporate insights from HR professionals to select relevant features.

### Model Training and Tuning

- **Support Vector Machines (SVM)**: The SVM model is trained using a radial basis function (RBF) kernel, known for handling non-linear relationships.
- **Hyperparameter Tuning**: Grid Search Cross-Validation is used to optimize the SVM parameters, ensuring the best model performance.

### Evaluation Metrics

- **Confusion Matrix**: To visualize the model's performance.
- **Accuracy, Precision, Recall**: Key metrics for evaluating the model.
- **ROC-AUC Score**: To assess the model's ability to distinguish between classes.

## Installation Guide

To replicate this project, clone the repository and install the required dependencies:

```bash
git clone https://github.com/HedyehRahmani/Prediction-Employee-Attrition.git
pip install -r requirements.txt
```

## Running the Project

After installation, you can run the project using the following command:

```bash
python app.py
```

This will preprocess the data, train the SVM model, and display the evaluation results.

## Results and Insights

The project provides a comprehensive analysis of employee attrition, highlighting the key factors that influence an employee's decision to leave. The SVM model offers robust predictions, with a focus on achieving a balance between precision and recall. Insights from this model can be used to inform HR policies and retention strategies, potentially saving the company significant costs.

## Contribution Guidelines

We welcome contributions from the community! If you have ideas for improving the model or adding new features, please fork the repository and submit a pull request. Contributions should align with the project's goal of enhancing predictive accuracy and providing actionable insights for HR professionals.
