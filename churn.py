#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 10:55:17 2018

@author: partha
"""

import pandas as pd
import numpy as np
pd.set_option('display.max_rows', 999)
pd.set_option('display.max_columns', 999)

%matplotlib inline
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import datetime as dt
from sklearn.metrics import matthews_corrcoef
df = pd.read_csv("/home/partha/Documents/Churn_Predict/churnpridiction.csv")
df.shape
label = pd.read_csv("/home/partha/Documents/Churn_Predict/Churned_List_with_details_09082018.csv")
df_all = df.merge(label[['subscriberid','Churned']],right_on='subscriberid',left_on='customerid',how='left')
df_all.isnull().sum()
df_all.drop(['Unnamed: 645','subscriberid'],axis=1,inplace=True)
df_all['customerlastseen'] = pd.to_datetime(df_all.customerlastseen)
#df_all['lastseendays'] = (max(df_all.customerlastseen) - df_all.customerlastseen)/np.timedelta64(1, 'D')
df_all.shape
list(df_all)
df_all.drop_duplicates(['customerid'],inplace=True)
df_all.isnull().sum()
df_all['Churned'] = np.where( df_all['Churned'] =='YES', 1,0)
X = df_all.drop(['Churned','customerid','customerlastseen'],axis=1,inplace=False)
X.dtypes
Y = df_all['Churned']
feature_name = X.columns.tolist()
#Y = np.where(y=='YES',1,0)
print ("Churned class 1: %s" %(Y.sum()*100.0/len(Y)))
print ("Not Churned class 0: %s" %((len(Y)-Y.sum())*100.0/len(Y)))


###################################### Feture Selection ######################################################################
categorical_list = []
numerical_list = []
for i in X.columns.tolist():
    if X[i].dtype=='object':
        categorical_list.append(i)
    else:
        numerical_list.append(i)
print('Number of categorical features:', str(len(categorical_list)))
print('Number of numerical features:', str(len(numerical_list)))

# 1.1 Pearson Correlation
def cor_selector(X, Y):
    cor_list = []
    # calculate the correlation with Y for each feature
    for i in X.columns.tolist():
        cor = np.corrcoef(X[i], Y)[0, 1]
        cor_list.append(cor)
    # replace NaN with 0
    cor_list = [0 if np.isnan(i) else i for i in cor_list]
    # feature name
    cor_feature = X.iloc[:,np.argsort(np.abs(cor_list))].columns.tolist()         # taking all feature
    #cor_feature = X.iloc[:,np.argsort(np.abs(cor_list))[-200:]].columns.tolist()  # taking only 200 feature
    # feature selection? 0 for not select, 1 for select
    cor_support = [True if i in cor_feature else False for i in feature_name]
    return cor_support, cor_feature

cor_support, cor_feature = cor_selector(X, Y)
print(str(len(cor_feature)), 'selected features')

# 1.2 Chi-square

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
X_norm = MinMaxScaler().fit_transform(X)
chi_selector = SelectKBest(chi2, k= 200)
chi_selector.fit(X_norm, Y)

chi_support = chi_selector.get_support()
chi_feature = X.loc[:,chi_support].columns.tolist()
print(str(len(chi_feature)), 'selected features')

#2. Wrapper
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=200, step=10, verbose=5)
rfe_selector.fit(X_norm, Y)

rfe_support = rfe_selector.get_support()
rfe_feature = X.loc[:,rfe_support].columns.tolist()
print(str(len(rfe_feature)), 'selected features')

# 3. Embeded
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

embeded_lr_selector = SelectFromModel(LogisticRegression(penalty="l1"), '1.25*median')
embeded_lr_selector.fit(X_norm, Y)

embeded_lr_support = embeded_lr_selector.get_support()
embeded_lr_feature = X.loc[:,embeded_lr_support].columns.tolist()
print(str(len(embeded_lr_feature)), 'selected features')

# 3.2 Random Forest
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

embeded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=100), threshold='1.25*median')
embeded_rf_selector.fit(X, Y)

embeded_rf_support = embeded_rf_selector.get_support()
embeded_rf_feature = X.loc[:,embeded_rf_support].columns.tolist()
print(str(len(embeded_rf_feature)), 'selected features')

# 3.3 LightGBM

from sklearn.feature_selection import SelectFromModel
from lightgbm import LGBMClassifier

lgbc=LGBMClassifier(n_estimators=643, learning_rate=0.05, num_leaves=32, colsample_bytree=0.2,reg_alpha=3, reg_lambda=1, min_split_gain=0.01, min_child_weight=40)

embeded_lgb_selector = SelectFromModel(lgbc, threshold='1.25*median')
embeded_lgb_selector.fit(X, Y)

embeded_lgb_support = embeded_lgb_selector.get_support()
embeded_lgb_feature = X.loc[:,embeded_lgb_support].columns.tolist()
print(str(len(embeded_lgb_feature)), 'selected features')

# Feature Summery
pd.set_option('display.max_rows', None)
# put all selection together
feature_selection_df = pd.DataFrame({'Feature':feature_name, 'Pearson':cor_support,'Chi-2':chi_support, 'RFE':rfe_support, 'Logistics':embeded_lr_support,'Random Forest':embeded_rf_support, 'LightGBM':embeded_lgb_support})
# count the selected times for each feature
feature_selection_df['Total'] = np.sum(feature_selection_df, axis=1)
# display the top 100
feature_selection_df = feature_selection_df.sort_values(['Total','Feature'] , ascending=False)
feature_selection_df.index = range(1, len(feature_selection_df)+1)
feature_selection_df.head(10)
feature_selection_df['feature'] = feature_selection_df['Total'].apply(filter(lambda x: feture_selection_df.Total[x] == 6) 


# Correlation plot between numerical values
#numeric_columns = list(X.columns[X.dtypes != 'category'])
#sns.pairplot(data = X, x_vars= numeric_columns, y_vars= numeric_columns, hue = 'Churn')

# heat map plot between numerical values
fig = plt.figure(figsize = (14,10))
corr = X[numerical_list].corr()
sns.heatmap(corr, mask = np.zeros_like(corr, dtype = np.bool), square = True,annot= True, cmap = sns.diverging_palette(220, 10, as_cmap= True))
plt.title("HeatMap between numerical columns of churn dataset")



# create training and testing vars

X = X[embeded_rf_feature]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
Y_train.sum()*100.0/Y_train.shape[0]
Y_test.sum()*100.0/Y_test.shape[0]
X_train = X_train.values
X_test = X_test.values
X_train.shape
seed=7
scoring = 'accuracy'

##============================== Load library for modelling ===================================================

import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_curve, auc
import warnings; warnings.simplefilter('ignore')
import numpy as np

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('RF', RandomForestClassifier()))
models.append(('XGB', GradientBoostingClassifier()))
models.append(('ADaBoost', AdaBoostClassifier()))
models.append(('SVM', SVC()))

##================== evaluate each model in turn===========================================
results = []
names = []

for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_test)

#Confusion Matrix
print(confusion_matrix(Y_test, predictions))
print(accuracy_score(Y_test, predictions)*100)
print(classification_report(Y_test, predictions))

# Make predictions on validation dataset
RF = RandomForestClassifier()
RF.fit(X_train, Y_train)
predictions = RF.predict(X_test)

#Confusion Matrix
print(confusion_matrix(Y_test, predictions))
print(accuracy_score(Y_test, predictions)*100)
print(classification_report(Y_test, predictions))

# Make predictions on validation dataset
XGB = GradientBoostingClassifier()
XGB.fit(X_train, Y_train)
predictions = XGB.predict(X_test)
# Make probability
predictions_1 = XGB.predict_proba(X_test)

# Function to calculate accuracy 
def cal_accuracy(Y_test, predictions): 
      
    print("Confusion Matrix: ", 
        confusion_matrix(Y_test, predictions)) 
      
    print ("Accuracy : ", 
    accuracy_score(Y_test,predictions)*100) 
      
    print("Report : ", 
    classification_report(Y_test, predictions)) 

#Confusion Matrix
print(confusion_matrix(Y_test, predictions))
print(accuracy_score(Y_test, predictions)*100)
print(classification_report(Y_test, predictions))

def PlotConfusionMatrix(y_test,predictions):
    cfn_matrix = confusion_matrix(Y_test,predictions)
    fig = plt.figure(figsize=(5,5))
    sns.heatmap(cfn_matrix,cmap='coolwarm_r',fmt='1',linewidths=0.5,annot=True)
    plt.title('Confusion Matrix')
    plt.ylabel('Real Classes')
    plt.xlabel('Predicted Classes')
    plt.show()
    print('---Classification Report---')
    print(classification_report(Y_test,predictions))
    print("ROC-AUC Score")
    print(roc_auc_score(Y_test,predictions))
    
PlotConfusionMatrix(Y_test,predictions)
plt.figure(figsize=(20,100))

import scikitplot as skplt
import matplotlib.pyplot as plt

skplt.metrics.plot_roc_curve(Y_test, predictions)
plt.show()
