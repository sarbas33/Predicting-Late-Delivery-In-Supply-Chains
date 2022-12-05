# Python script to automate Model development for predicting late delivery risk in a supply chain

#--------->info : logger_file is a variable used to log all the information about various model building stages into a log file.
#--------->Dotnet application which calls this automation script recieves the status of the model through print statements.
#--------->If first 6 characters of the line printed is 'status', then the text which follows will be shown in the User Interface Application inside the status bar.
#--------->If first 6 characters of the line printed is 'result', then the following value will be shown in the results table of the Application.


#start by creating a log file in the output directory. Log files will help in tracing the status of a particular run.
print("status: Started Python automation script")
logger_file=open("log_file.txt","w")
logger_file.writelines("started \n")
logger_file.writelines("importing  libraries")
print("status: Importing  libraries ")

#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

# Get Inputs from the Application
data_location=sys.argv[1]
target_col =sys.argv[2]
cardinality_limit=sys.argv[3]
corr_matrix_threshold=sys.argv[4]
variance_threshold=sys.argv[5]

#read dataset to a dataframe
logger_file.writelines("Reading dataset \n")
print("status: Reading dataset")
input_dataset_raw_form=pd.read_csv(data_location,encoding="ISO-8859-1")
num_columns=input_dataset_raw_form.shape[1]
logger_file.writelines("Imported dataset. Number of columns is "+ num_columns + " \n" )
print("status: Imported dataset.")

# fill null values 
input_dataset_raw_form.fillna(value = 0, inplace = True)

# import more librararies for data analysis
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Remove target column from input columns list
logger_file.writelines("setting input columns \n")
print("status: Setting input columns ")
input_cols = input_dataset_raw_form.columns.tolist().remove(target_col)

#*********************************************feature engineering*********************************************************************************
# create a df for input data
input_data=input_dataset_raw_form[input_cols]
corr_data=input_dataset_raw_form[input_cols+[target_col]]

#divide input columns to numerical and categorical
logger_file.writelines("Categorize input columns to numerical and categorical \n")
print("status: Categorizing input columns to numerical and categorical ")
numeric_cols =input_data.select_dtypes(include=np.number).columns.tolist()
categorical_cols = input_data.select_dtypes('object').columns.tolist()

#some columns may have small number of unique values. They can be directly one-hot encoded.
logger_file.writelines("Group categorical columns into small and big \n")
print("status: Grouping categorical columns into small and big")
categorical_small=[]
categorical_big=[]
for i in categorical_cols:
    if input_dataset_raw_form[i].nunique() <= cardinality_limit:
        categorical_small.append(i)
    else:
        categorical_big.append(i)
logger_file.writelines("Grouping categorical to small and big completed \n")
print("status: Grouping categorical to small and big completed")

# human analysis can help to remove some features from the dataset, input_dataset_after_data_cleaning is exactly the same input_dataset_raw_form if there is no such intervention. 
# Further work can focus in removing unwanted features using certain algorithms, in that case the cleaned dataset can be named as input_dataset_after_data_cleaning
input_dataset_after_data_cleaning=input_dataset_raw_form[input_cols+[target_col]]

# create a df with numerical columns
numeric_cols_data=input_data[numeric_cols]

# Converting numerical features to same scale
logger_file.writelines("Converting numerical features to same scale \n")
print("status: Converting numerical features to same scale")
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(input_dataset_after_data_cleaning[numeric_cols])
input_dataset_after_data_cleaning[numeric_cols] = scaler.transform(input_dataset_after_data_cleaning[numeric_cols])
logger_file.writelines("Converting numerical features to same scale completed \n")
print("status: Converting numerical features to same scale completed")

# encoding categorical variables
logger_file.writelines("one-hot encoding started(for small number of unique values) \n")
print("status: one-hot encoding started(for small cardinality features)")
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
encoder.fit(input_dataset_after_data_cleaning[categorical_small])
encoded_cols = list(encoder.get_feature_names(categorical_small))
input_dataset_after_data_cleaning[encoded_cols] = encoder.transform(input_dataset_after_data_cleaning[categorical_small])
logger_file.writelines("one-hot encoding completed \n")
print("status: one-hot encoding completed ")

#target guided encoding for large number of cardinality
logger_file.writelines("target_guided_encoding started \n")
print("status: target_guided_encoding started")
from sunbird.categorical_encoding import target_guided_encoding
for i in categorical_big:
    target_guided_encoding(input_dataset_after_data_cleaning, i,target_col)
logger_file.writelines("target_guided_encoding completed \n")
print("status: target_guided_encoding completed ")

corr_1=corr_data.corr()
encoded_cols=encoded_cols+categorical_big


# **************************************************************feature selection*************************************************************

#chi square test
logger_file.writelines("chi2 test started \n")
print("status: chi2 test started ")
from sklearn.feature_selection import chi2
X = input_dataset_after_data_cleaning[encoded_cols]
y = input_dataset_after_data_cleaning[target_col]
chi_scores = chi2(X,y)
p_values =p_values_1= pd.Series(chi_scores[1],index = X.columns)
categorical_selected=[]
for i in range(len(p_values_1)):
    if p_values_1[i]<0.1:
        categorical_selected.append(encoded_cols[i])
categorical_selected

logger_file.writelines("chi2 test completed \n")
print("status:chi2 test completed ")

#feature selection for numerical variables.
logger_file.writelines("creating correlation matrix \n")
print("status: creating correlation matrix ")
plt.figure(figsize=(10,10))
sns.heatmap(corr_1, cbar=True, square=True, fmt='.2f', annot=True,annot_kws={'size':8}, cmap='Blues')
upper_tri =corr_1.where(np.triu(np.ones(corr_1.shape),k=1).astype(np.bool))
corr_matrix_threshold=0.95
correlated = [column for column in upper_tri.columns if any(upper_tri[column] > corr_matrix_threshold)]
print(correlated)
logger_file.writelines("removing correlated columns \n")
print("status: removing correlated columns")
numeric_after_corr=[]
for i in numeric_cols:
    if i not in correlated:
        numeric_after_corr.append(i)

#variance threshold method
logger_file.writelines("VarianceThreshold start \n")
print("status: VarianceThreshold started ")
from sklearn.feature_selection import VarianceThreshold
vt = VarianceThreshold(threshold=variance_threshold)
df_var=input_dataset_after_data_cleaning[numeric_after_corr]
transformed = vt.fit_transform(df_var)
_ = vt.fit(df_var)
mask = vt.get_support()
df_var_reduced = df_var.loc[:, mask]
df_var_reduced.shape
i=df_var_reduced.columns
df_var_reduced.columns
numeric_cols_selected=[]
for x in i:
    numeric_cols_selected.append(x)
logger_file.writelines("Feature selection for Numerical columns completed \n")
print("status: Feature selection for Numerical columns completed  ")


# combine selected numerical and categorical columns
input_cols_final=categorical_selected
for i in numeric_cols_selected:
    input_cols_final.append(i)



#*************************************model building****************************************************************************


recall_list=[]
logger_file.writelines(" train test split for catboost  start \n")
print("status: train test split for catboost  started")
from sklearn.model_selection import train_test_split
train_val_df, test_df = train_test_split(input_dataset_raw_form, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=42)

train_inputs = train_df[input_cols].copy()
train_targets = train_df[target_col].copy()
test_inputs=test_df[input_cols].copy()
test_targets = test_df[target_col].copy()

logger_file.writelines(" train test split for catboost end \n")
print("status: train test split for catboost ended")

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn import metrics

#build models using catboost
if num_columns<30:
    logger_file.writelines("catboost model building started \n")
    print("status: catboost model building started")
    import catboost
    from catboost import CatBoostClassifier
    model=CatBoostClassifier()
    model.fit(train_inputs,train_targets,cat_features=categorical_cols)
    test_preds_catboost = model.predict(test_inputs)
    recall=recall_score(test_targets, test_preds_catboost)
    recall_list.append(recall)
    logger_file.writelines("catboost model building end \n")
    print("status: catboost model building end")
else:
    recall_list.append(0)
    print("status: catboost model building skipped")

train_val_df, test_df = train_test_split(input_dataset_after_data_cleaning, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=42)

print('train_df.shape :', train_df.shape)
print('val_df.shape :', val_df.shape)
print('test_df.shape :', test_df.shape)

train_inputs = train_df[input_cols_final].copy()
train_targets = train_df[target_col].copy()

val_inputs = val_df[input_cols_final].copy()
val_targets = val_df[target_col].copy()

test_inputs = test_df[input_cols_final].copy()
test_targets = test_df[target_col].copy()

train_inputs.head()


#Build model using Logistic regression 
logger_file.writelines("Logistic Regression model building start \n")
print("status: Logistic Regression model building started")
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='liblinear')
model.fit(train_inputs, train_targets)

X_train = train_inputs[input_cols_final]
X_val = val_inputs[input_cols_final]
X_test = test_inputs[input_cols_final]
train_preds_logistic = model.predict(X_train)

train_probs_logistic = model.predict_proba(X_train)
test_preds_logistic = model.predict(X_test)
recall=recall_score(test_targets, test_preds_logistic)
recall_list.append(recall)
logger_file.writelines("Logistic Regression model building end \n")
print("status: Logistic Regression model building completed")

#Build model using GaussianNB
logger_file.writelines("GaussianNB model building strat \n")
print("status: GaussianNB model building strated")

from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(train_inputs, train_targets)
train_preds_gaussian = model.predict(X_train)
train_probs_gaussian = model.predict_proba(X_train)
test_preds_gaussian = model.predict(X_test)
recall=recall_score(test_targets, test_preds_gaussian)
recall_list.append(recall)
logger_file.writelines("GaussianNB model building end \n")
print("status: GaussianNB model building completed")

#Build model using RandomForest
logger_file.writelines("RandomForestClassifier model building strat \n")
print("status: RandomForestClassifier model building strated")
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=50)#doubt
model.fit(train_inputs, train_targets)
train_preds_random = model.predict(X_train)
train_probs_random= model.predict_proba(X_train)
test_preds_random = model.predict(X_test)
recall=recall_score(test_targets, test_preds_random)
recall_list.append(recall)

logger_file.writelines("RandomForestClassifier model building end \n")
print("status: RandomForestClassifier model building completed")

#Build model using XGBoost
logger_file.writelines("xgboost model building strat \n")
print("status: xgboost model building strated")
import xgboost as xgb
from xgboost import XGBClassifier
model=xgb.XGBClassifier()
model.fit(train_inputs, train_targets)
train_preds_xgb = model.predict(X_train)
train_probs_xgb = model.predict_proba(X_train)
test_preds_xgb = model.predict(X_test)
recall=recall_score(test_targets, test_preds_xgb)
recall_list.append(recall)

logger_file.writelines("xgboost model building completed \n")
print("status: xgboost model building completed")

#Build model using LightGBM
logger_file.writelines("lightgbm model building strat \n")
print("status: lightgbm model building strated")
import lightgbm as lgb
from lightgbm import LGBMClassifier
model = lgb.LGBMClassifier(learning_rate=0.09,max_depth=-5,random_state=42)
model.fit(train_inputs, train_targets)
train_preds_lgb = model.predict(X_train)
train_probs_lgb = model.predict_proba(X_train)
test_preds_lgb = model.predict(X_test)
recall=recall_score(test_targets, test_preds_lgb)
recall_list.append(recall)
logger_file.writelines("lightgbm model building completed \n")
print("status: lightgbm model building completed")


#******************************************Display Performance Metrics of best model in the UI****************************************

#find the model with best performance
logger_file.writelines("finding best model and getting results  \n")
print("status: finding best model and getting results")
index=recall_list.index(max(recall_list))
#send all results of the best model to application
if index==0:
    recall_score=recall_score(test_targets, test_preds_catboost)
    accuracy_score=accuracy_score(test_targets, test_preds_catboost)
    f1_score=f1_score(test_targets, test_preds_catboost)
    precision_score=precision_score(test_targets, test_preds_catboost)
    roc_auc_score=roc_auc_score(test_targets, test_preds_catboost)

    print("result1","Catboost")
    print("result2",recall_score)
    print("result3",accuracy_score)
    print("result4",f1_score)
    print("result5",precision_score)
    print("result6",roc_auc_score)
   
    rf=CatBoostClassifier()
    rf.fit(test_inputs, test_targets)
    metrics.plot_roc_curve(rf,X_test,test_targets) 

elif index==1:
    recall_score=recall_score(test_targets, test_preds_logistic)
    accuracy_score=accuracy_score(test_targets, test_preds_logistic)
    f1_score=f1_score(test_targets, test_preds_logistic)
    precision_score=precision_score(test_targets, test_preds_logistic)
    roc_auc_score=roc_auc_score(test_targets, test_preds_logistic)

    print("result1","Logistic Regression")
    print("result2",recall_score)
    print("result3",accuracy_score)
    print("result4",f1_score)
    print("result5",precision_score)
    print("result6",roc_auc_score)
    
    rf=LogisticRegression(solver='liblinear',multi_class='ovr')
    rf.fit(test_inputs, test_targets)
    metrics.plot_roc_curve(rf,X_test,test_targets)
    
elif index==2:
    recall_score=recall_score(test_targets, test_preds_gaussian)
    accuracy_score=accuracy_score(test_targets, test_preds_gaussian)
    f1_score=f1_score(test_targets, test_preds_gaussian)
    precision_score=precision_score(test_targets, test_preds_gaussian)
    roc_auc_score=roc_auc_score(test_targets, test_preds_gaussian)

    print("result1","Gaussin NB")
    print("result2",recall_score)
    print("result3",accuracy_score)
    print("result4",f1_score)
    print("result5",precision_score)
    print("result6",roc_auc_score)
    
    rf=GaussianNB()
    rf.fit(test_inputs, test_targets)
    metrics.plot_roc_curve(rf,X_test,test_targets)
elif index==3:
    recall_score=recall_score(test_targets, test_preds_random)
    accuracy_score=accuracy_score(test_targets, test_preds_random)
    f1_score=f1_score(test_targets, test_preds_random)
    precision_score=precision_score(test_targets, test_preds_random)
    roc_auc_score=roc_auc_score(test_targets, test_preds_random)

    print("result1","Random Forest")
    print("result2",recall_score)
    print("result3",accuracy_score)
    print("result4",f1_score)
    print("result5",precision_score)
    print("result6",roc_auc_score)
    
    rf=RandomForestClassifier(n_estimators=50)
    rf.fit(test_inputs,test_targets)
    metrics.plot_roc_curve(rf,X_test,test_targets)
elif index==4:
    recall_score=recall_score(test_targets, test_preds_xgb)
    accuracy_score=accuracy_score(test_targets, test_preds_xgb)
    f1_score=f1_score(test_targets, test_preds_xgb)
    precision_score=precision_score(test_targets, test_preds_xgb)
    roc_auc_score=roc_auc_score(test_targets, test_preds_xgb)

    print("result1","XG Boost")
    print("result2",recall_score)
    print("result3",accuracy_score)
    print("result4",f1_score)
    print("result5",precision_score)
    print("result6",roc_auc_score)
    
    rf=XGBClassifier()
    rf.fit(test_inputs, test_targets)
    metrics.plot_roc_curve(rf,X_test,test_targets)
elif index==5:
    recall_score=recall_score(test_targets, test_preds_lgb)
    accuracy_score=accuracy_score(test_targets, test_preds_lgb)
    f1_score=f1_score(test_targets, test_preds_lgb)
    precision_score=precision_score(test_targets, test_preds_lgb)
    roc_auc_score=roc_auc_score(test_targets, test_preds_lgb)

    print("result1","Lightgbm")
    print("result2",recall_score)
    print("result3",accuracy_score)
    print("result4",f1_score)
    print("result5",precision_score)
    print("result6",roc_auc_score)
    
    rf=LGBMClassifier()
    rf.fit(test_inputs, test_targets)
    metrics.plot_roc_curve(rf,X_test,test_targets)
    
logger_file.write("Model Building Completed")
print("status: Model Building Completed")
logger_file.close()