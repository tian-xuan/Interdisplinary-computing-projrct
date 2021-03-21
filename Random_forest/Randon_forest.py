import numpy as np
#import sklearn
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFECV
import joblib
import matplotlib.pyplot as plt
#%%
'''
read data
'''
data = pd.read_csv("../Data/Train_data.csv")

X = data.iloc[:,0:8] #Input argument
Y = data.iloc[:,8] #Output argument

#data.dropna(inplace = True) #if there is any NA block
'''
Split data into train and test group with 0.9 and 0.1 randomly
'''
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1,test_size=0.1) #n_splits for only split once
split.get_n_splits(X,Y)
for train_ind,test_ind in split.split(X,Y):
    X_train, X_test = X.iloc[train_ind],X.iloc[test_ind]
    Y_train, Y_test = Y.iloc[train_ind],Y.iloc[test_ind]
#%%
'''
Creat different combinations of features to give a better training result
'''
poly = PolynomialFeatures()#default = 2
X_train = poly.fit_transform(X_train)
feature_name = poly.get_feature_names(data.columns)#show the combinations
print(feature_name)

'''
Select the above combination of features by run the random forest technique once 
and abandon those low weight arguments.
'''
select = SelectFromModel(RandomForestClassifier(),max_features = 30)#maximum feature to retain is 30
select = select.fit(X_train,Y_train)
featuresSupport = select.get_support()
X_train = select.transform(X_train)
'''
find the selected features' name
'''
selecred_feature_name = []
for i in range(len(featuresSupport)):
    if featuresSupport[i] == True:
        selecred_feature_name.append(feature_name[i])
print(selecred_feature_name)
#%%
'''
Creat many random forest model with different split and leaf size, 
run each forest with some sample data, and leave the best performed one
'''
gs_b = GridSearchCV(
    param_grid = {'min_samples_leaf':np.linspace(5,55,10).astype(int),
                  'min_samples_split':np.linspace(5,55,10).astype(int)},
    estimator = RandomForestClassifier(n_estimators=1000),
    scoring = 'accuracy')#The selection of the best model is based on the accuracy, other candidate are 'f1', 'balanced_accuracy', 'roc_auc'
'''
Train the model
'''
gs_b.fit(X_train,Y_train)

# recursive feature elimination and cross-calidated selection of best number of features.
gsFeatureSelector = RFECV(gs_b.best_estimator_, cv = 5).fit(X_train,Y_train)
gsX = gsFeatureSelector.transform(X_train)
gsFeatureSupport = gsFeatureSelector.support_

gs_a = GridSearchCV(
    param_grid = {'min_samples_leaf':np.linspace(5,55,10).astype(int),
                  'min_samples_split':np.linspace(5,55,10).astype(int)},
    estimator = RandomForestClassifier(n_estimators=1000),
    scoring = 'accuracy')
gs_a.support = gsFeatureSupport
gs_a.selector = gsFeatureSelector
gs_a.fit(gsX,Y_train)#Train the model

#%%
# Linear Regression model and feature selection
linearRegression = LinearRegression()
linearFeatureSelector = RFECV(linearRegression, cv = 5).fit(X_train,Y_train)
LinearX = linearFeatureSelector.transform(X_train)
linearFeatureSupport = linearFeatureSelector.support_
# store selector in Linear Regression Model
linearRegression.support = linearFeatureSupport
linearRegression.selector = linearFeatureSelector

# train Linear Regression Model
linearRegression.fit(LinearX,Y_train)

# %%

'''
We can choose to open or not open a model
'''
#with open('\Model_RandomForest.joblib', 'rb') as gs_a:

#gs_a = joblib.load('./Model_RandomForest.joblib')


"""
Verification
"""
verifi_data = pd.read_csv("../Data/Verification_data.csv")

X_ver = verifi_data.iloc[:,0:8] #Input argument
Y_ver = verifi_data.iloc[:,8] #Output argument

X_ver = poly.fit_transform(X_ver)#Generate test feature
X_ver = select.transform(X_ver)#Selecr test feature

X_gs = gs_a.selector.transform(X_ver)
X_linear = linearRegression.selector.transform(X_ver)

print(gs_a.score(X_gs,Y_ver))#print the accuracy of the random forest result
print(linearRegression.score(X_linear,Y_ver))#print the accuracy of the linear regression result

result = gs_a.predict(X_gs)
np.savetxt('RandomForest_Prediction.txt',result,fmt = '%d') #if we want to export our prediction
print(gs_a.predict(X_gs))# print the prediction of each row


weight = gs_a.best_estimator_.feature_importances_
print(gs_a.best_estimator_.feature_importances_)#print the statistical weight of each feature in the forest
#%%
'''
save the model if we want
'''
#joblib.dump(gs_a, 'Model_RandomForest.joblib')


#%% 
'''
Visulisation part starts from here
'''
#%%
'''
Statistical Weight diagram (bar chart)
'''
#Bar chart
params = {
    'axes.labelsize': 10,
    'font.size':22,
    'legend.fontsize': 22,
    'xtick.labelsize': 10,
    'ytick.labelsize': 20,
    'figure.figsize': [30,8]
    } 
plt.rcParams.update(params)
plt.rcParams["font.family"]="Times New Roman"

fig1, ax1 = plt.subplots()
ax1.bar(3*np.linspace(1, len(weight),len(weight)),weight, color = 'blue')
ax1.set_xticks(3*np.linspace(1, len(weight),len(weight)))
ax1.set_xticklabels(selecred_feature_name)
ax1.set(ylim = [0,0.2])








