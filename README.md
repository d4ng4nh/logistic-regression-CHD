# Heart Disease Prediction Model (Project for MAT1201)

## Overview
This project builds a machine learning model to predict the likelihood of heart disease within the next 10 years based on various health attributes from the **Framingham Heart Study** dataset. The model uses **logistic regression** and includes data cleaning, outlier removal, and oversampling techniques to improve performance.

## Key Features
- **Data Cleaning**: Handles missing values by filling with median or zero.
- **Outlier Removal**: Uses the IQR method to remove outliers.
- **SMOTE**: Balances the dataset by oversampling the minority class (heart disease).
- **Modeling**: Logistic regression is used to predict heart disease risk.

## Dataset
- The dataset includes features like age, gender, smoking habits, blood pressure, BMI, glucose levels, and other health conditions.
- The target variable is `TenYearCHD`, indicating whether the person has heart disease (1) or not (0).

## Requirements
Install the required libraries:
```bash
pip install pandas scikit-learn matplotlib seaborn imbalanced-learn
