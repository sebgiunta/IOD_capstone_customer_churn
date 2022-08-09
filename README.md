# Customer Churn Prediction (IOD): Project Overview 
* Created predictive model that predicts whether a customer will churn (F1 0.9274) to help banks intervene and prevent churn.
* Data sourced from kaggle (https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers)
* Cleaned data and encoded relevant variables for model building
* Compared a range of state-of-the-art Machine Learning Models (CatBoost, XGBoost, AdaBoost, Random Forest)
* Optimised CatBoost using GridSearchCV to reach the best model
* Consolidated information into PowerPoint Slide Deck for presentation.

## Code and Resources Used 
**Python Version:** 3.8  
**Packages:** pandas, numpy, sklearn, matplotlib, seaborn, catboost, xgboost, imblearn, custom code

## Data Cleaning
The data sourced from Kaggle was very clean. Pre-processing was required, which was done prior to model building.

## EDA
EDA was done on the individual columns, to get a sense of counts and distributions. Additionally, the target variable was explored in more detail. Some highlights below:

![alt text](https://github.com/sebgiunta/iod_capstone_customer_churn/blob/main/images/categorical.png "Categorical Variables")
![alt text](https://github.com/sebgiunta/iod_capstone_customer_churn/blob/main/images/numerical.png "Numerical Variables")
![alt text](https://github.com/sebgiunta/iod_capstone_customer_churn/blob/main/images/target.png "Target Variable")

## Model Building 
First, I encoded the categorical variables. I also split the data into train and tests sets with a test size of 30%. StandardScaler was used on the data and implemented through use of a pipeline.

I tried 11 different models and evaluated them using F1 Score. I chose F1 as it is a useful metric for imbalanced data.

Three sampling techniques were tested (RandomOverSampler, RandomUnderSampler, SMOTE). RandomOverSample gave the best performance.

5-Fold Cross Validation was done on the top 4 performing models (CatBoost, XGBoost, Random Forest, AdaBoost).

## Model performance
The CatBoost model outperformed the other approaches on the test and validation sets. 
*	**CatBoost** : F1 = 0.9207
*	**XGBoost**: F1 = 0.9121
*	**Random Forest**: F1 = 0.8730
*	**AdaBoost**: F1 = 0.8294

Hyperparameter tuning increased the CatBoost F1 score to 0.9274

## Feature Importance
Below shows the importance of each feature

![alt text](https://github.com/sebgiunta/iod_capstone_customer_churn/blob/main/images/target.png "Feature Importance")
