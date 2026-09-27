# Predictive Maintenance and Machine Failure Type Classification

## Project Overview

This project uses Machine Learning to predict the type of machine failure based on machine operating conditions.

The project uses the AI4I 2020 Predictive Maintenance Dataset and applies a Random Forest Classifier for multiclass failure type classification.

## Objectives

- Analyze machine operating data.
- Clean and preprocess the dataset.
- Identify different machine failure types.
- Handle class imbalance using SMOTE.
- Train a Random Forest classification model.
- Perform hyperparameter tuning using RandomizedSearchCV.
- Evaluate the model using accuracy, precision, recall, F1-score, and confusion matrix.
- Save the trained model using Pickle.
- Create a Streamlit web application for making predictions on new machine data.

## Dataset

The project uses the AI4I 2020 Predictive Maintenance Dataset.

The main input features used by the model are:

- Product Type
- Air Temperature [K]
- Process Temperature [K]
- Rotational Speed [rpm]
- Torque [Nm]
- Tool Wear [min]

The target variable is:

- No Failure
- TWF
- HDF
- PWF
- OSF
- RNF

Rows containing multiple simultaneous failure types were excluded for multiclass classification.

## Machine Learning Process

The following steps were performed:

1. Dataset loading
2. Data cleaning
3. Identification of failure types
4. Feature selection
5. Train-test split
6. Categorical feature encoding
7. Random Forest classification
8. Handling class imbalance using SMOTE
9. Model evaluation
10. Hyperparameter tuning using RandomizedSearchCV
11. Confusion matrix analysis
12. Model saving using Pickle
13. Prediction on new machine data

## Models Used

### Random Forest

A Random Forest Classifier was initially trained on the original training data.

### Random Forest with SMOTE

SMOTE was applied to the training data to improve the representation of minority failure classes.

### Tuned Random Forest

RandomizedSearchCV was used to search for suitable Random Forest hyperparameters.

## Model Performance

The tuned Random Forest model achieved approximately:

- Accuracy: 96.14%
- Macro F1-score: 0.58

The Random Forest with SMOTE achieved approximately:

- Accuracy: 96.34%
- Macro F1-score: 0.59

Accuracy is not considered alone because the dataset contains a large imbalance between normal machines and failure classes.

## Prediction

The trained model was saved as:

`predictive_maintenance_model.pkl`

The model can be loaded and used to predict the failure type of a new machine based on its operating conditions.

## Streamlit Application

A Streamlit web application was developed using `app.py`.

The application allows the user to enter:

- Product Type
- Air Temperature
- Process Temperature
- Rotational Speed
- Torque
- Tool Wear

and predicts the corresponding machine failure type.

## How to Run the Project

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL