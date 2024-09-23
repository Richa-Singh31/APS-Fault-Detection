# Air Pressure System - Fault Detection

## Problem Statement
The Air Pressure System generates pressurized air that is utilized in various functions in heavy duty vehicles, such as braking and gear changes. The dataset' positive class corresponds to the truck failures for a specific component of the APS System. The negative class corresponds to the truck with failures not related to the APS System. The problem is to reduce the cost due to unnecessary repairs. So it is required to minimize the false predictions.

## Dataset
The dataset used is publicly available and can be downloaded through the provided link:

https://www.kaggle.com/datasets/uciml/aps-failure-at-scania-trucks-data-set


## Key Components
- Data Ingestion: The Data Ingestion component extracts data from MongoDB, processes it into a feature store, and splits it into training and test datasets for machine learning model development.
- Data Validation: The Data Validation component verifies the integrity of the data by checking for schema conformity, missing numerical columns, and detecting dataset drift between training and testing datasets.
- Data Transformation: The Data Transformation component preprocesses the data by imputing missing values, scaling features, handling class imbalance, and generating transformed training and testing datasets for machine learning.
- Model Trainer: The Model Trainer component trains the machine learning model, evaluates its performance, and ensures it meets accuracy and overfitting thresholds before saving the trained model for deployment.
- Model Evaluation: The Model Trainer component trains the machine learning model, evaluates its performance, and ensures it meets accuracy and overfitting thresholds before saving the trained model for deployment.
- Model Pusher: The Model Pusher component saves the trained model to specified locations, making it ready for deployment and future use.

## Install Dependencies
```bash
pip install -r requirements.txt
```
## Run
```bash
python main.py
```

## Conclusion

This project implements a machine learning pipeline to tackle the challenges of the Air Pressure System (APS) in heavy-duty vehicles, focusing on accurately predicting truck failures to minimize unnecessary repair costs. It encompasses data ingestion, validation, and transformation to ensure data quality and optimal model performance. The pipeline trains models that effectively distinguish between APS-related failures and others, aiming to reduce false predictions. Through thorough model evaluation, the best-performing model is deployed for practical use.
