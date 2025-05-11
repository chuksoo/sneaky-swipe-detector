import os
import sys
import shap
import mlflow
import numpy as np
import pandas as pd
from ipywidgets import widgets
from catboost import CatBoostClassifier
from datetime import datetime
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn import set_config
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from imblearn.datasets import make_imbalance
from imblearn.over_sampling import RandomOverSampler
pd.options.mode.chained_assignment = None # to avoid SettingWithCopyWarning after scaling
  
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier 
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier 
from xgboost import XGBClassifier 

from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

import warnings
warnings.filterwarnings('ignore')
print('Project libraries has been successfully been imported!')


# read the dataset using the compression zip
transactions = pd.read_json('https://github.com/CapitalOneRecruiting/DS/blob/master/transactions.zip?raw=true', compression='zip', lines=True)
 
print('Data has been read correctly!')


# In[3]:


# generate profile for transactions data
profile = ProfileReport(transactions, title="Data profile report for transactions data", html={'style': {'full_width': True}})
profile



# change data to the right type
def change_datatype(df, cols, type_val):
    for col in cols:
        df[col] = df[col].astype(type_val)
        
change_datatype(transactions, ['transactionDateTime', 'currentExpDate', 'accountOpenDate', 'dateOfLastAddressChange'], 'datetime64[ns]')
change_datatype(transactions, ['accountNumber', 'customerId', 'creditLimit', 'cardCVV', 'enteredCVV', 'cardLast4Digits'], 'int32')
change_datatype(transactions, ['availableMoney', 'transactionAmount', 'currentBalance'], 'float32')


# function to make new features 
def create_features(data):
    data['transactionMonth'] = data['transactionDateTime'].dt.month_name()
    data['transactionDayofWeek'] = data['transactionDateTime'].dt.day_name()
    data['transactionHour'] = data['transactionDateTime'].dt.hour
    data['transactionMinutes'] = data['transactionDateTime'].dt.minute
    data['transactionSeconds'] = data['transactionDateTime'].dt.second
    data['currentExpMonth'] = data['currentExpDate'].dt.month_name()
    data['currentExpDayofWeek'] = data['currentExpDate'].dt.day_name()
    data['accountOpenMonth'] = data['accountOpenDate'].dt.month_name()
    data['accountOpenDayofWeek'] = data['accountOpenDate'].dt.day_name()
    data['dateOfLastAddressChangeMonth'] = data['dateOfLastAddressChange'].dt.month_name()
    data['dateOfLastAddressChangeDayofWeek'] = data['dateOfLastAddressChange'].dt.day_name()
    
# create new feature
create_features(transactions)
change_datatype(transactions, ['transactionHour', 'transactionMinutes', 'transactionSeconds'], 'int32') 


# In[8]:


# check data information
transactions.info()



# correlation matrix
numeric_data = transactions.select_dtypes(include=['int64', 'int32', 'float64', 'float32'])
corr_matrix = numeric_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=False)
plt.title('Correlation Matrix for transactions data')
plt.show()


# <div id="question_two">
#     <h3>Question 2: Plot</h3> 
# </div>

# In[13]:


# histogram of the processed amounts of each transaction
fig, ax = plt.subplots(figsize=(8, 6))
sns.histplot(transactions, x='transactionAmount', kde=True, bins=50).set_title('Histogram of processed amounts of each transaction');





# identify reversed transaction
transactions['isDuplicated'] = (transactions.sort_values(['transactionDateTime'])
                                .groupby(['customerId', 'transactionAmount'], sort=False)['transactionDateTime']
                                .diff()
                                .dt.total_seconds()
                                .lt(120)
                               )


# In[16]:


# check for duplicate transactions
transaction_duplicate = transactions.query('isDuplicated == True')
transaction_duplicate.head()


# In[17]:


# what type of purchases exist
transaction_duplicate.transactionType.unique()


# In[18]:


# check for reversed transactionAmounts
n = ["REVERSAL"]
reversed_df = transaction_duplicate.loc[transaction_duplicate.transactionType.isin(n)]
reversed_df.shape


# In[19]:


# total dollar amount reversed
reversal_amount = reversed_df.transactionAmount.sum()
print('The total number of transactions with reversal was {} transactions and the total dollar amount reversed was ${:.2f}'.format(reversed_df.shape[0], reversal_amount))


# Here we can see that *`3824`* transactions had a reversal and this amounted to a total dollar amount of *`$562,291.88`*. Let's check for multi-swipe transactions that occured.

# In[20]:


# check for multi-swipe that occured
n = ['PURCHASE', 'ADDRESS_VERIFICATION', '']
multi_swipe_df = transaction_duplicate.loc[transaction_duplicate.transactionType.isin(n)]
multi_swipe_df.shape


# In[21]:


# total count of multi-swipe
multi_swipe_count = multi_swipe_df.shape[0]

# total dollar amount for multi-swipe
multi_swipe_amount = multi_swipe_df['transactionAmount'].sum()

print('The total number of transactions with multi-swipe is {}, and total estimated dollar amount for the multi-swipe transactions is ${:.2f}'.format(multi_swipe_count, multi_swipe_amount))


# For multi-swipe transactions, we can see that *`5058`* transactions were multi-swiped and this amounted to a total dollar amount of *`$742,053.50`*. 

# In[22]:


# top merchant with reversed transaction amounts
reversed_df.merchantName.value_counts().reset_index().rename(columns = {'index': 'merchantName', 'merchantName': 'Count'}).head(5)


# In[23]:


plot_snsbar(reversed_df, 'merchantName', 'transactionAmount', 'Top 20 Merchant with reversed transaction amounts')


# In[24]:


# top 20 merchant with most multi-swipe activities
plot_snsbar(multi_swipe_df, 'merchantName', 'transactionAmount', 'Top 20 Merchant with most multi-swipe activities')


# From the above plots, we can see that `Lyft`, `Uber`, `Old Navy`, `Gap`, and `Amazon` were among the top five merchants with reversed transaction amount. Similarly, `Uber`, `Lyft`, `Alibaba.com`, `Apple`, and `Ebay` had the most multi-swipe activities occurring. 

# ### Drop multi-swipe transactions

# In[25]:


# drop duplicated transactions from the data
rows = transaction_duplicate.index
card_transactions = transactions.drop(index = rows)
card_transactions.shape


# In[26]:


# determine the number of fraud cases in the dataset
fraud = card_transactions[card_transactions['isFraud'] == True]
valid = card_transactions[card_transactions['isFraud'] == False]
print(f'The total number of fraudulent transactions are {len(fraud)} while the number of good transactions are {len(valid)}') 


# ### Analyze data

# In[27]:


# check day of the week most fraud cases occur
fraud.transactionDayofWeek.value_counts().reset_index().rename(columns = {'index': 'transactionDayofWeek', 'transactionDayofWeek': 'Count'})


# We can see that most fraudulent transactions took place on `Friday`, `Monday` and `Wednesday`. Although we should note that fraudulent activities are distributed evenly among the days of the week. Let's check the month with the most fraud taking place. 

# In[28]:


# check month when most fraud cases occur
fraud.transactionMonth.value_counts().reset_index().rename(columns = {'index': 'transactionMonth', 'transactionMonth': 'Count'})


# The month of `May`, `January` and `March` had the most fraudulent transactions occuring. 

# In[29]:


# check top 20 merchant that had these fraudulent activities
fraud.merchantName.value_counts().reset_index().rename(columns = {'index': 'merchantName', 'merchantName': 'Count'}).head(20)


# In[30]:


# count of fraudulent activities across merchants
plot_snsbar(fraud, 'merchantName', 'transactionAmount', 'Top 20 Merchant with the most fraudulent transactions')


# The plot above shows the top 20 merchants with the most fraudulent activities occuring.

# In[31]:


# what activities took place when these frauds occured
plot_snsbar(fraud, 'merchantCategoryCode', 'transactionAmount', 'Top 20 activities that resulted into fraudulent transactions')


# The chart above shows the activities that took place when these fraudulent activities occured. We can see that `online retail`, `online gifts`, `rideshare`, `fastfood`, `food`, and `entertainment` accounted for the top activities that usually resulted in a fraudulent transactions.

# In[32]:


# pie chart showing fraud activity carried out without card presnt
merchant_fraud_agg = fraud[['merchantCategoryCode', 'cardPresent', 'transactionAmount']]
merchant_fraud_agg_without_card = merchant_fraud_agg[merchant_fraud_agg['cardPresent'] == False]
merchant_fraud_without_card_present = merchant_fraud_agg_without_card.groupby(['merchantCategoryCode'])['transactionAmount'].count().reset_index().sort_values(by='transactionAmount', ascending = False).reset_index(drop = True)
pie_chart(merchant_fraud_without_card_present, 'Fraudulent activities carried out without card')


# The pie chart above shows the fraction of fraudulent activities that was carried out without the card being present. `Online retail` and `online gifts` accounted for the most representation.

# In[33]:


# pie chart showing fraud activity carried out with card presnt
merchant_fraud_agg = fraud[['merchantCategoryCode', 'cardPresent', 'transactionAmount']]
merchant_fraud_agg_with_card = merchant_fraud_agg[merchant_fraud_agg['cardPresent'] == True]
merchant_fraud_with_card_present = merchant_fraud_agg_with_card.groupby(['merchantCategoryCode'])['transactionAmount'].count().reset_index().sort_values(by='transactionAmount', ascending = False).reset_index(drop = True)
pie_chart(merchant_fraud_with_card_present, 'Fraudulent activities carried out with card present')


# The pie chart above shows the fraction of fraudulent activities that was carried out with the card being present. `Fast food`, `food` and `entertainment` accounted for the most representation.

# ### Conclusion
# 
# In this section, we explored the data in order to derive insight from the data. We plotted a histogram of `transactionAmount` and noticed that the plot is *skewed left* with the spread of the data from 0 to 2011 with a mean of 136.98 and a median at 87.9. On further exploration of the data, `3824` transactions had a reversal and this amounted to a total dollar amount of `$562,291.88`. About `5058` transactions were multi-swiped and this amounted to a total dollar amount of `$742,053.50`. We noted that most fraudulent transactions took place on `Friday`, `Monday` and `Wednesday` and the months of `May`, `January` and `March` were the top three months where the most fraudulent transactions occured.
# 
# We decided to locate the merchants with the most reversed transaction. We observe that `Lyft`, `Uber`, `Old Navy`, `Gap`, and `Amazon` were among the top five merchants with reversed transaction amount. Similarly, `Uber`, `Lyft`, `Alibaba.com`, `Apple`, and `Ebay` had the most multi-swipe activities occurring. We observed activities that took place when these fraudulent activities occured and noticed that `online retail`, `online gifts`, `rideshare`, `fastfood`, `food`, and `entertainment` accounted for the top activities that usually resulted in a fraudulent transactions. The fraction of fraudulent activities that was carried out without the card being present shows that `Online retail` and `online gifts` accounted for the most frauds. Also, the fraction of fraudulent activities that was carried out with the card present shows that `fast food`, `food` and `entertainment` accounted for the most representation.
# 
# Intuitively, we can deduce from the data that most of the fraud being carried out without the card present could represent real fraud that would lead to losses for CapitalOne because you really need your card present to make purchases online. One way to prevent this type of online fraud would be to prevent online transaction from going through if the card is not present with the owner.

# <div id="modeling_process">
#     <h2>Modeling Process</h2> 
# </div>

# <div id="question_four">
#     <h3>Question 4: Model</h3> 
# </div>

# Here, we would train different models. We would be training a couple of tree-based models, gradient-boosted model and leaf-based model. The primary metric we chose to evaluate the model is AUC-ROC. The secondary metric is accuracy. AUC computes the area under the curve and the objective is to **maximize** this area. Accuracy tells us how often the classifier is correct and the objective is to **maximize** accuracy. 

# ### Feature Engineering for Machine Learning
# 
# We perform feature engineering to encode all categorical features to numeric. Encoding features makes them useful for machine learning. We would be applying one-hot encoding, target encoding and ordinal encoding depending on the machine learning algorithm. A summary of the result is shown here.
# 
# | Model type | Model | Encoding type | Highlight | Cons |
# |:--- |:----|:---:|:---:| :--- |
# | Statistical based| Logistic regression | One-hot encoding | Less prone to over-fitting and easily explainable | Can overfit in high dimensional datasets |
# | Tree-based | Decision Tree | label encoding | Normalization or scaling of data not needed  | Prone to overfitting   |
# |            | Random Forest | label encoding | Excellent predictive powers| Prone to overfitting   |
# | Gradient-boosted | Catboost      | No encoding    | Can handle categorical data well |  Needs to build deep decision trees in features with high cardinality |
# | Gradient boosted | XGBoost | One-hot encoding | Good execution and model performance | Cannot handle categorical features (need encoding) |
# | Gradient-boosted | LightGBM | Ordinal encoding | Extremely fast | Needs encoding for categorical features    |
# 
# For ease of modeling, we would use OrdinalEncoder for features with high cardinality and OneHotEncoder for features with low cardinality. We chose this approach to avoid problems associated with the curse of dimensionality. Since the CatBoost regressor has its own implementation for encoding of categorical features, we create a separate dataset without any encoding. Internally, catboost encodes the categorical features. Since one-hot encoding is not optimal for the LightGBM since it does its own encoding for categorical features, we apply OrdinalEncoder on the categorical features of LightGBM.

# In[34]:


# Get sample of the data without replacement 
transactions_df = card_transactions.sample(frac=0.5, replace=False, random_state=42)

# drop unimportant features
transactions_df = transactions_df.drop(['transactionDateTime', 'currentExpDate', 'accountOpenDate', 'dateOfLastAddressChange', 'cardCVV', 'enteredCVV', 
                                        'cardLast4Digits', 'echoBuffer', 'merchantCity', 'merchantState', 'merchantZip', 'posOnPremises', 'recurringAuthInd', 'isDuplicated'], axis=1)

# encode some features
transactions_df['expirationDateKeyInMatch'] = transactions_df['expirationDateKeyInMatch'].replace({True: 1, False: 0})
transactions_df['cardPresent'] = transactions_df['cardPresent'].replace({True: 1, False: 0})
transactions_df['isFraud'] = transactions_df['isFraud'].replace({True: 1, False: 0})

# create copy of dataset for CatBoost algorithm 
card_transactions_catBoost = transactions_df.copy()

# create features and target
features = transactions_df.drop(['isFraud'], axis = 1)
target = transactions_df.isFraud


# ### Preprocessing Pipeline

# In[35]:


# Get numerical and categorical features
cat_feature_cols = [cname for cname in features.columns if features[cname].dtype == "object"]
num_feature_cols = [cname for cname in features.columns if features[cname].dtype in ["int64", "int32", "float64", "float32"]] 

# preprocessing pipelines
num_pipeline = Pipeline(
    steps=[("scaler", StandardScaler())
    ]
)

cat_pipeline = Pipeline(
    steps=[('encoding', OrdinalEncoder())
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num_pipeline", num_pipeline, num_feature_cols),
        ("cat_pipeline", cat_pipeline, cat_feature_cols)
        ]
    )

# show preprocessor pipeline
set_config(display="diagram")
display(preprocessor)


# In[36]:


# Apply all the stages of transformation to the data
preprocessed_data = preprocessor.fit_transform(features)

# Get feature names after transformation
preprocessed_features = pd.DataFrame(preprocessed_data, columns=num_feature_cols + cat_feature_cols)
preprocessed_features.head()


# ### Split data into 60% training, 20% validation and 20% testing sets
# 
# Here we split the data into training, validation and testing sets in the ratio 60:20:20 respectively.

# In[37]:


# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(preprocessed_features, target, test_size=0.20, random_state=12345)

# split train data into validation and train 
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.25, random_state = 12345) # 0.25 * 0.80 = 0.20 for validation size

# display the shape of the split dataset
print('The train set now contains {}'.format(X_train.shape[0]) + ' dataset representing 60% of the data') 
print('The valid set now contains {}'.format(X_valid.shape[0]) + ' dataset representing 20% of the data')
print('The test set now contains {}'.format(X_test.shape[0]) + ' dataset representing 20% of the data')


# ### Conclusion
# 
# We split the data into 60% training, 20% validation and 20% testing sets. We did this because we had a lot of data and validating the model is equally as important as training the model. We applied ordinal encoding to encode the categorical features. We scaled the data after encoding using the standard scaler function. Next we are going to examine class imbalance and apply some oversampling techniques to improve the model performance if class imbalance exist.

# ### Examine Class imbalance

# In[38]:


# function to calculate model evaluation metrics
def print_model_evaluation(y_test, test_predictions):
    print("\033[1m" + 'F1 score: ' + "\033[0m", '{:.3f}'.format(f1_score(y_test, test_predictions)))
    print("\033[1m" + 'Accuracy Score: ' + "\033[0m", '{:.2%}'.format(accuracy_score(y_test, test_predictions)))
    print("\033[1m" + 'Precision: ' + "\033[0m", '{:.3f}'.format(precision_score(y_test, test_predictions)))
    print("\033[1m" + 'Recall: ' + "\033[0m", '{:.3f}'.format(recall_score(y_test, test_predictions)))
    print("\033[1m" + 'Balanced Accuracy Score: ' + "\033[0m", '{:.2%}'.format(balanced_accuracy_score(y_test, test_predictions)))
    print("\033[1m" + 'AUC-ROC Score: ' + "\033[0m", '{:.2%}'.format(roc_auc_score(y_test, test_predictions)))
    print()
    print("\033[1m" + 'Confusion Matrix' + "\033[0m")
    print('-'*50)
    print(confusion_matrix(y_test, test_predictions))
    print()
    print("\033[1m" + 'Classification report' + "\033[0m")
    print('-'*50)
    print(classification_report(y_test, test_predictions))
    print()


# #### Baseline Model

# In[40]:


# baseline model using a dummy classifier
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)
dummy_clf_valid_predictions = dummy_clf.predict(X_valid)


# In[41]:


# evaluate baseline model
print_model_evaluation(y_valid, dummy_clf_valid_predictions)


# The baseline model predicts the most frequent class in this case to be "0". Looking at the baseline model report, we can see that the accuracy is high at 98.42% and the AUC-ROC score is 50%. This represents the baseline so we should expect our models to perform better. We need to balance the classes to improve on the model's performance. Let's perform a sanity check using Logistic regression.

# #### Sanity check with Logistic regression

# In[42]:


# sanity check
model = LogisticRegression(random_state=12345, solver='liblinear')
model.fit(X_train, y_train) # train the model 
valid_predictions = pd.Series(model.predict(X_valid))
class_frequency = valid_predictions.value_counts(normalize=True)
print(class_frequency)
class_frequency.plot(kind='bar');
print()
print('Accuracy score before upsampling: {:.3f}'.format(accuracy_score(y_valid, valid_predictions)))


# We assess the sanity of the model by checking how often the target feature contains the class "1" or "0". We can observe the class imbalance in the predicted validation set. Next we try to improve the quality of the model using the upsampling approaches or applying SMOTE to fix class imbalance. Since the classes are highly imbalanced, we would use the ROC-AUC, Log loss and F1 score as our metric of choice. 

# ### 

# ### Fix Class Imbalance

# In[43]:


# Check class distribution
class_frequency = y_train.value_counts() #(normalize = True)
print(class_frequency)
class_frequency.plot(kind='bar')
plt.title('Class Frequency before resampling')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.show()


# In[44]:


# Random Over-Sampling with Imblearn
oversample = RandomOverSampler(sampling_strategy="all", random_state=42)
X_train_oversample, y_train_oversample = oversample.fit_resample(X_train, y_train)
print("Original dataset shape", Counter(y_train))
print("Resampled dataset shape", Counter(y_train_oversample))


# In[45]:


# Check class distribution after resampling
class_frequency = y_train_oversample.value_counts() #(normalize = True)
print(class_frequency)
class_frequency.plot(kind='bar')
plt.title('Class Frequency after resampling')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.show()


# This wasn't too bad. We have been able to balance the class. Now it's time to build our model.

# <div id="model_training">
#     <h2>Model Training</h2> 
# </div>

# In[46]:


# function to plot confusion matrix
def plot_confusion_matrix(y, y_predict):
    "Plots the confusion matrix"
    cm = confusion_matrix(y, y_predict)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax, fmt='g'); #annot=True to annotate cells
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(['is not Fraud', 'isFraud']); ax.yaxis.set_ticklabels(['is not Fraud', 'isFraud'])

# Get model performance score
def get_scores(y_test, y_pred, model):
    # determine AUC-ROC score
    print("\033[1m" + 'Measure model performance using {} Classifier'.format(model) + "\033[0m")
    print('ROC-AUC Score: {:.3f}'.format(roc_auc_score(y_test, y_pred[1], multi_class='ovr')))
    print('Log loss is: {:.3f}'.format(log_loss(y_test, y_pred[1])))
    print('F1 score: {:.3f}'.format(f1_score(y_test, y_pred[0], average="weighted")))

def shap_feature_importance(model_name, model, X_train):
    # compute SHAP values
    shap.initjs()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train, approximate = False)

    class_names = [str(cls) for cls in model.classes_]
    shap.summary_plot(shap_values, X_train.values, show = False, plot_type="bar", class_names = class_names, feature_names = X_train.columns) 
    fig, ax = plt.gcf(), plt.gca()
    fig.set_size_inches(8, 6)

    # Modifying main plot parameters
    ax.tick_params(labelsize=14)
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=10)
    ax.set_title('Feature Importance - {} model on features'.format(model_name), fontsize=12)

    # Modifying color bar parameters
    ax.tick_params(labelsize=10)
    ax.set_ylabel("Feature value", fontsize=10)
    plt.show()




# #### Logistic Regression

# In[47]:


get_ipython().run_cell_magic('time', '', '\n# define hyperparameters to tune \ngrid_logreg = {\n    "C": [0.01, 0.1, 1],\n    \'penalty\': [\'l2\'], # l1 lasso l2 ridge\n    \'solver\': [\'lbfgs\']\n}\nkfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)\n\n# define the model\nlogreg_clf = LogisticRegression(random_state = 12345)\n# define the grid search object\ngrid_search_logreg = GridSearchCV(\n    estimator = logreg_clf,\n    param_grid = grid_logreg,\n    scoring = \'roc_auc\',\n    cv = kfold\n)\n# execute search\nlogreg_cv = grid_search_logreg.fit(X_train_oversample, y_train_oversample)\n# summarize result\nprint(\'The best hyperparameters are: {}\'.format(logreg_cv.best_params_))\n')


# In[48]:


# train model and make predictions
def train_logistic_regression_classifier(X_train, y_train):
    """This function trains a logistic regression classifier"""
    # build the model
    logreg_model = LogisticRegression(**grid_search_logreg.best_params_)
    logreg_model.fit(X_train, y_train) # train the model 
    return logreg_model
    
def logistic_regression_classifier_prediction(X_train, y_train, X_test, y_test):
    """
    This function is used to make prediction 
    using a logistic regression classification model
    """
    logreg_model = train_logistic_regression_classifier(X_train, y_train)
    logreg_pred = logreg_model.predict(X_test)
    logreg_pred_proba = logreg_model.predict_proba(X_test)[:, 1]
    return logreg_pred, logreg_pred_proba


# In[49]:


get_ipython().run_cell_magic('time', '', '\n# train classifier and make prediction\nlogreg_pred = logistic_regression_classifier_prediction(X_train_oversample, y_train_oversample, X_valid, y_valid)\n\n# get predictions\nget_scores(y_valid, logreg_pred, "Logistic regression")\n')


# In[50]:


# Plot feature importance for Logistic Regression
logreg_model = train_logistic_regression_classifier(X_train_oversample, y_train_oversample)
# get importance for logistic regression
log_model_importance = pd.DataFrame(
    logreg_model.coef_[0], index=X_train_oversample.columns, columns=["Importance"]
)
# plot the chart
log_model_importance.sort_values(by="Importance").plot(kind="bar", figsize=(12, 6))
plt.xticks(rotation=45, ha="right", rotation_mode="anchor", fontsize=13)
plt.xlabel("Logistic Regression Classifier Feature Importance")
plt.show()


# #### Decision Tree Classifier

# In[51]:


get_ipython().run_cell_magic('time', '', '# hyperparameter optimization for Decision tree classifier\n\n# define hyperparameters to tune\ngrid_dt = {\n    "criterion" : ["gini", "entropy"],\n    "max_depth" : [None, 2, 4, 8, 10, 12],\n    "min_samples_split" : [2, 4, 8, 16],\n    "min_samples_leaf" : [2, 4, 6]\n}\nkfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)\n\n# define the model \ndt_clf = DecisionTreeClassifier(random_state = 12345)\n# define the grid search\ngrid_search_dt = GridSearchCV(estimator=dt_clf, \n                              param_grid = grid_dt, \n                              cv = kfold, \n                              scoring = \'roc_auc\')\n# execute search\ndt_cv = grid_search_dt.fit(X_train_oversample, y_train_oversample)\n# summarize result\nprint(\'The best hyperparameters are: {}\'.format(dt_cv.best_params_))\n')


# In[52]:


# train the model
def train_decision_tree_classifier(X_train, y_train):
    """This function trains a Decision Tree classifier"""
    # build the model
    dt_model = DecisionTreeClassifier(**grid_search_dt.best_params_)
    dt_model.fit(X_train, y_train) # train the model 
    return dt_model
    
def decision_tree_classifier_prediction(X_train, y_train, X_test, y_test):
    """
    This function is used to make prediction 
    using the Decsion Tree classifier
    """
    dt_model = train_decision_tree_classifier(X_train, y_train)
    dt_pred = dt_model.predict(X_test)
    dt_pred_proba = dt_model.predict_proba(X_test)[:, 1]
    return dt_pred, dt_pred_proba


# In[53]:


get_ipython().run_cell_magic('time', '', '\n# train classifier and make prediction\ndt_pred = decision_tree_classifier_prediction(X_train_oversample, y_train_oversample, X_valid, y_valid)\n\n# get predictions\nget_scores(y_valid, dt_pred, "Decision Tree")\n')


# In[54]:


# Plot feature importance for Decision Tree
dt_model = train_decision_tree_classifier(X_train_oversample, y_train_oversample)
shap_feature_importance("Decision Tree", dt_model, X_train_oversample)


# #### Random Forest Classifier

# In[55]:


get_ipython().run_cell_magic('time', '', '# hyperparameter optimization\n\n# define hyperparameters to tune\ngrid_rf = {\n    "n_estimators" : [10, 25, 50, 100],\n    "max_depth" : [2, 4, 8, 10, 12],\n    "min_samples_leaf" : [2, 4, 6]\n}\nkfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)\n\n# define the model \nrf_clf = RandomForestClassifier(random_state = 12345)\n# define the grid search\ngrid_search_rf = GridSearchCV(estimator=rf_clf, \n                              param_grid = grid_rf, \n                              cv=kfold, \n                              scoring = \'roc_auc\')\n# execute search\nrf_cv = grid_search_rf.fit(X_train_oversample, y_train_oversample)\n# summarize result\nprint(\'The best hyperparameters are: {}\'.format(rf_cv.best_params_))\n')


# In[56]:


# train the model
def train_random_forest_classifier(X_train, y_train):
    """This function trains a random forest classifier"""
    # build the model
    rf_model = RandomForestClassifier(**grid_search_rf.best_params_)
    rf_model.fit(X_train, y_train) # train the model 
    return rf_model
    
def random_forest_classifier_prediction(X_train, y_train, X_test, y_test):
    """
    This function is used to make prediction 
    using the random forest  model
    """
    rf_model = train_random_forest_classifier(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]
    return rf_pred, rf_pred_proba


# In[57]:


get_ipython().run_cell_magic('time', '', '\n# train classifier and make prediction\nrf_pred = random_forest_classifier_prediction(X_train_oversample, y_train_oversample, X_valid, y_valid)\n\n# get predictions\nget_scores(y_valid, rf_pred, "Random Forest")\n')


# #### XGBoost Classifier

# In[58]:


get_ipython().run_cell_magic('time', '', '# hyperparameter optimization for XGBoost Classifier\n\n# define hyperparameters to tune\ngrid_xgb = {\'learning_rate\': [0.01, 0.1, 0.3], \n            \'max_depth\': [2, 4, 6, 10],\n            \'n_estimators\': [50, 100, 200, 500]\n           }\nkfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)\n\n# define the model \nxgb_model = XGBClassifier(random_state = 12345, verbosity = 0)\n# define the grid search\ngrid_search_xgb = GridSearchCV(\n    estimator = xgb_model, \n    param_grid = grid_xgb, \n    scoring = "roc_auc", \n    cv = kfold, \n    n_jobs = 1\n)\n# execute search\nxgb_cv = grid_search_xgb.fit(X_train_oversample, y_train_oversample)\n# summarize result\nprint(\'The best hyperparameters are: {}\'.format(xgb_cv.best_params_))\n')


# In[59]:


# train the model
def train_xgboost_classifier(X_train, y_train):
    """This function trains an XGBoost classifier"""
    # build the model
    xgb_model = XGBClassifier(**grid_search_xgb.best_params_)
    xgb_model.fit(X_train, y_train) # train the model 
    return xgb_model
    
def xgboost_classifier_prediction(X_train, y_train, X_test, y_test):
    """
    This function is used to make prediction 
    using the XGBoost classifier model
    """
    xgb_model = train_xgboost_classifier(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
    return xgb_pred, xgb_pred_proba


# In[60]:


get_ipython().run_cell_magic('time', '', '\n# train classifier and make prediction\nxgb_pred = xgboost_classifier_prediction(X_train_oversample, y_train_oversample, X_valid, y_valid)\n\n# get predictions\nget_scores(y_valid, xgb_pred, "XGBoost")\n')


# #### LightGBM Classifier

# In[61]:


get_ipython().run_cell_magic('time', '', "# hyperparameter optimization for LightGBM classifier\n\n# define hyperparameters to tune\ngrid_lgbm = {'learning_rate': [0.001, 0.01, 0.05, 0.1],\n             'n_estimators': [50, 100, 200],\n             'num_leaves': [5, 10, 20, 31],\n             'verbose': [0]\n            }\nkfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)\n\n# define the model \nlgbm_clf = LGBMClassifier(random_state = 12345) #, verbosity = 0)\n# define the grid search\ngrid_search_lgbm = GridSearchCV(\n    estimator = lgbm_clf, \n    param_grid = grid_lgbm, \n    scoring = 'roc_auc', \n    cv = kfold, \n    n_jobs = 1\n)\n# execute search\nlgbm_cv = grid_search_lgbm.fit(X_train_oversample, y_train_oversample)\n# summarize result\nprint('The best hyperparameters are: {}'.format(lgbm_cv.best_params_))\n")


# In[62]:


# train the model
def train_lightgbm_classifier(X_train, y_train):
    """This function trains a LightGBM classifier"""
    # build the model
    lgbm_model = LGBMClassifier(**grid_search_lgbm.best_params_)
    lgbm_model.fit(X_train, y_train) # train the model 
    return lgbm_model
    
def lightgbm_classifier_prediction(X_train, y_train, X_test, y_test):
    """
    This function is used to make prediction 
    using the LightGBM classifier model
    """
    lgbm_model = train_lightgbm_classifier(X_train, y_train)
    lgbm_pred = lgbm_model.predict(X_test)
    lgbm_pred_proba = lgbm_model.predict_proba(X_test)[:, 1]
    return lgbm_pred, lgbm_pred_proba


# In[86]:


get_ipython().run_cell_magic('time', '', '\n# train classifier and make prediction\nlgbm_pred = lightgbm_classifier_prediction(X_train_oversample, y_train_oversample, X_valid, y_valid)\n\n# get predictions\nget_scores(y_valid, lgbm_pred, "LightGBM")\n')


# #### CatBoost Classifier

# In[87]:


# let's balance classes for the CatBoost
 
# create features and target
X = card_transactions_catBoost.drop(['isFraud'], axis = 1)
y = card_transactions_catBoost.isFraud

# split data into training and testing sets
features_train, features_test, target_train, target_test = train_test_split(X, y, test_size=0.20, random_state=12345)
print(features_train.shape, target_train.shape, features_test.shape)

# split train data into validation and train 
X_train_cb, X_valid_cb, y_train_cb, y_valid_cb = train_test_split(features_train, target_train, test_size = 0.25, random_state = 12345) # 0.25 * 0.80 = 0.20 for validation size
print(X_train_cb.shape, y_train_cb.shape, X_valid_cb.shape)

# resample training data for catboost
categorical_cols = [cname for cname in X_train_cb.columns if X_train_cb[cname].dtype == 'object']  # categorical columns 
categorical_column_index = [X_train_cb.columns.get_loc(c) for c in categorical_cols if c in X_train_cb] # categorical column index
numeric_cols = [cname for cname in X_train_cb.columns if X_train_cb[cname].dtype in ["int64", "int32", "float64", "float32"]] 

# apply random oversampling to train data
oversampler = RandomOverSampler(sampling_strategy="minority", random_state=42)
X_train_oversample_cb, y_train_oversample_cb = oversampler.fit_resample(X_train_cb, y_train_cb)
print(X_train_oversample_cb.shape, y_train_oversample_cb.shape)

# features scaling
scaler = StandardScaler()
scaler.fit(X_train_oversample_cb[numeric_cols])
# transform the train, validation and test set
X_train_oversample_cb[numeric_cols] = scaler.transform(X_train_oversample_cb[numeric_cols])
X_valid_cb[numeric_cols] = scaler.transform(X_valid_cb[numeric_cols])
features_test[numeric_cols] = scaler.transform(features_test[numeric_cols])

# display the shape of the split dataset for CatBoost method
print()
print('The train set now contains {}'.format(X_train_oversample_cb.shape[0]) + ' dataset representing 60% of the data') 
print('The valid set now contains {}'.format(X_valid_cb.shape[0]) + ' dataset representing 20% of the data')
print('The test set now contains {}'.format(features_test.shape[0]) + ' dataset representing 20% of the data')


# In[88]:


get_ipython().run_cell_magic('time', '', '# hyperparameter optimization for catboost\n\n# define hyperparameters to tune\ngrid_cb = {\'learning_rate\': [0.001, 0.01, 0.5],\n        \'depth\': [4, 6, 10],\n        \'l2_leaf_reg\': [1, 3, 5, 7, 9]\n       }\nkfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)\n\n# define the model \ncatboost_clf = CatBoostClassifier(\n    iterations=200,\n    cat_features = categorical_cols,\n    logging_level = \'Silent\',\n    eval_metric=\'AUC\',\n    early_stopping_rounds = 50,\n    random_state = 12345)\n# define the grid search\ngrid_search_cb = GridSearchCV(estimator = catboost_clf, param_grid = grid_cb, scoring="roc_auc", cv=kfold)\n# execute search\ncatboost_cv = grid_search_cb.fit(X_train_cb, y_train_cb)\n# summarize result\nprint(\'The best hyperparameters are: {}\'.format(catboost_cv.best_params_))\n')


# In[89]:


# function to train model and make predictions
def train_catboost_classifier(X_train, y_train, X_test, y_test):
    """This function trains a catboost classifier model"""
    # build the model
    cb_model = CatBoostClassifier(**grid_search_cb.best_params_)
    cb_model.fit(X_train, y_train, cat_features = categorical_cols, eval_set=(X_test, y_test), verbose=False, plot=False) # train the model 
    return cb_model
    
def catboost_classifier_prediction(X_train, y_train, X_test, y_test):
    """
    This function is used to make prediction 
    using the catboost classifier model
    """
    cb_model = train_catboost_classifier(X_train, y_train, X_test, y_test)
    cb_pred = cb_model.predict(X_test)
    cb_pred_proba = cb_model.predict_proba(X_test)[:, 1]
    return cb_pred, cb_pred_proba


# In[90]:


get_ipython().run_cell_magic('time', '', '\n# train classifier and make prediction\ncb_pred = catboost_classifier_prediction(X_train_oversample_cb, y_train_oversample_cb, X_valid_cb, y_valid_cb)\n\n# get predictions\nget_scores(y_valid_cb, cb_pred, "CatBoost")\n')


# #### Conclusion
# 
# We tuned, trained and made predictions using six models. The LightGBM seems to be the fastest algorithm. We chose the XGBoost as the best performing model and use it for testing the test dataset.

# <div id="model_analysis">
#     <h2>Model Analysis</h2> 
# </div>

# #### Conclusion
# 
# Here, different classification models were trained and tested. A summary of the models, their respective AUC-ROC and accuracy score, the time it took to train and test the model is shown below. The best model that performed well on the training dataset is the XGBoost classifier. We used the result from this model as our final model.
# 
# |Models | Hyperparameter tuning time | Training time | ROC-AUC score | Log loss | F1 score|
# |:----|:-----:|:-----|:-----|:-----|:-----|
# | Logistic Regression | 4min 17s | 32.8s | 0.697 | 0.642 | 0.813 |
# | Decision Tree Classifier| 10min 40s | 3.97s | 0.533 | 1.044 | 0.969 |
# | Random Forest Classifier | 26min 1s | 40.4s | 0.798 | 0.396 | 0.907 |
# | XGBoost Classifier  | 32min 18s | 56.4s | 0.773 | 0.110 | 0.980 |
# | LightGBM Classifier | 6min 6s | 5.05s | 0.816 | 0.384 | 0.894 |
# | CatBoost Classifier | 3h 10min 23s | 12min 13s | 0.754 | 0.142 | 0.963 |
# 

# Here we choose the best model that performs best on the training dataset. We would be analyzing the speed and quality of the trained models.

# In[91]:


# determine best algorithm
models = {'LogisticRegression':logreg_cv.best_score_,
          'DecisionTree':dt_cv.best_score_,
          'RandomForest':rf_cv.best_score_,
          'XGBoost': xgb_cv.best_score_,
          'LightGBM': lgbm_cv.best_score_,
          'CatBoost': catboost_cv.best_score_
         }

bestalgorithm = max(models, key=models.get)
print('Best model is', bestalgorithm,'with a score of', models[bestalgorithm])
if bestalgorithm == 'LogisticRegression':
    print('Best params is :', logreg_cv.best_params_)
if bestalgorithm == 'DecisionTree':
    print('Best params is :', dt_cv.best_params_)
if bestalgorithm == 'RandomForest':
    print('Best params is :', rf_cv.best_params_)
if bestalgorithm == 'XGBoost':
    print('Best params is :', xgb_cv.best_params_)
if bestalgorithm == 'LightGBM':
    print('Best params is :', lgbm_cv.best_params_)
if bestalgorithm == 'CatBoost':
    print('Best params is :', catboost_cv.best_params_)


# #### ROC Curve for Classification Models

# In[94]:


# Fit multiple models and plot AUC ROC curve for binary classification
logreg_model = train_logistic_regression_classifier(X_train_oversample, y_train_oversample)
dt_model = train_decision_tree_classifier(X_train_oversample, y_train_oversample)
rf_model = train_random_forest_classifier(X_train_oversample, y_train_oversample)
xgb_model = train_xgboost_classifier(X_train_oversample, y_train_oversample)
lgbm_model = train_lightgbm_classifier(X_train_oversample, y_train_oversample)
catboost_model = train_catboost_classifier(X_train_oversample_cb, y_train_oversample_cb, X_valid_cb, y_valid_cb)

def plot_roc_curve(models, X_train, y_train, X_test, y_test):
    plt.figure(figsize=(8, 6))  # Set the size of the plot

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Compute the ROC curve and AUC for binary classification
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f"{name}, AUC = {roc_auc:.4f}")

    # Plot the random classifier line
    plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve for Binary Classification')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

# Define the model dictionary
models = {
    "Logistic regression": logreg_model,
    "Random Forest": rf_model,
    "Decision Tree": dt_model,
    "XGBoost": xgb_model,
    "LightGBM": lgbm_model,
    "CatBoost": catboost_model
}

# Call the plot_roc_curve function with encoded class labels
plot_roc_curve(models, X_train_oversample, y_train_oversample, X_valid, y_valid)


# <div id="model_testing">
#     <h2>Model Testing</h2> 
# </div>

# The XGBoost classifier is chosen as the best performing model for the final testing for this task because it meets our metrics goal of maximizing ROC-AUC score and F1 score, and minimizing the log loss. Using this model, we evaluate on the test dataset.

# In[95]:


# Make prediction with LightGBM
xgb_pred = xgboost_classifier_prediction(X_train_oversample, y_train_oversample, X_test, y_test)

# get predictions
get_scores(y_test, xgb_pred, "XGBoost")


# In[110]:


# Plot feature importance for XGBoost Classifier
shap_feature_importance("XGBoost", xgb_model, X_train_oversample)


# In[111]:


# plot confusion matrix for predictions
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
class_labels = ['isNotFraud', 'isFraud']
cm = confusion_matrix(y_test, xgb_pred[0], labels=xgb_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
disp.plot()
plt.yticks(rotation=90)
plt.title('Confusion Matrix for XGBoost Model Predictions')
plt.show()


