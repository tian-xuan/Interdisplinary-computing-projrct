import numpy as np
#import sklearn
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
import joblib
import matplotlib.pyplot as plt
#%%
'''
read data
'''
data = pd.read_csv("Data/Train_data.csv")

X = data.iloc[:,0:8] #Input argument
Y = data.iloc[:,8] #Output argument

#data.dropna(inplace = True)
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
'''
Creat many random forest model with different split and leaf size, 
run each forest with some sample data, and leave the best performed one
'''
gs = GridSearchCV(
    param_grid = {'min_samples_leaf':np.linspace(5,55,10).astype(int),
                  'min_samples_split':np.linspace(5,55,10).astype(int)},
    estimator = RandomForestClassifier(n_estimators=1000),
    scoring = 'accuracy')#The selection of the best model is based on the accuracy, other candidate are 'f1', 'balanced_accuracy', 'roc_auc'

'''
Train the model
'''
gs.fit(X_train,Y_train)#Train the model

# %%
"""
Varification
"""
varifi_data = pd.read_csv("Data/Verification_data.csv")

X_ver = data.iloc[:,0:8] #Input argument
Y_ver = data.iloc[:,8] #Output argument

X_ver = poly.fit_transform(X_ver)#Generate test feature
X_ver = select.transform(X_ver)#Selecr test feature


print(gs.score(X_ver,Y_ver))#print the accuracy of the result


print(gs.predict(X_ver))# print the prediction of each row


weight = gs.best_estimator_.feature_importances_
print(gs.best_estimator_.feature_importances_)#print the statistical weight of each feature in the forest
#%%
'''
save the model if we want
'''
#joblib.dump(gs, 'model')




#%% 
'''
Visulisation part starts from here
'''
#%%
'''
Statistical Weight diagram (bar chart or pie chart)
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

# Pie chart
params = {
    'axes.labelsize': 20,
    'font.size':22,
    'legend.fontsize': 22,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'figure.figsize': [20,18]
    } 
plt.rcParams.update(params)
plt.rcParams["font.family"]="Times New Roman"
# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = selecred_feature_name
sizes = weight
#explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig2, ax2 = plt.subplots()
ax2.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=False, startangle=90)
ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
#%% Plot for accuracy

Correct = 0
Wrong = 0

for i in range(len(Y_test)):
    if np.array(Y_test)[i] == gs.predict(X_test)[i]:
        Correct+=1
    else:
        Wrong+=1

params = {
    'axes.labelsize': 20,
    'font.size':22,
    'legend.fontsize': 22,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'figure.figsize': [10,8]
    }
plt.rcParams.update(params)
plt.rcParams["font.family"]="Times New Roman"
 
fig3, ax3 = plt.subplots()
explode = (0, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')
ax3.pie([Correct, Wrong], explode=explode, labels=['Correct','Wrong'], autopct='%1.1f%%',
        shadow=False, startangle=90)







