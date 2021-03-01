# %%
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit

# %%
data = pd.read_csv("diabetes.csv")

X = data.iloc[:,0:8] #Input argument
Y = data.iloc[:,8] #Output argument

split = StratifiedShuffleSplit(n_splits=1,test_size=0.1) #n_splits for only split once
split.get_n_splits(X,Y)
for train_ind,test_ind in split.split(X,Y):
    X_train, X_test = X.iloc[train_ind],X.iloc[test_ind]
    Y_train, Y_test = Y.iloc[train_ind],Y.iloc[test_ind]

Train_data = pd.concat([X_train,Y_train],axis=1) 
Verification_data = pd.concat([X_test,Y_test],axis=1)

Train_data.to_csv('Train_data.csv',index=False)
Verification_data.to_csv('Verification_data.csv',index=False)
# %%

