# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import compare_model as cm

params = {
   'axes.labelsize': 18,
   'font.size': 12,
   'legend.fontsize': 12,
   'xtick.labelsize': 12,
   'ytick.labelsize': 12,
   } 
plt.rcParams.update(params)

# %%
# import the result predicted from the model here 
prediction = np.loadtxt('../Random_forest/RandomForest_Prediction(1).txt')


# %%
'''
Show the verification data
'''
# inport the verification data
ver_data = list(np.loadtxt("../Data/Verification_data.csv",dtype=int,delimiter=',',skiprows=1,usecols=(8)))

# classify the verification dataset
p_0,p_1 = cm.binary_distribution(ver_data)


# %%
'''
Check with random guess
'''
# use a random guess to make prediction
# in theory the sensitivity should be p_1 and the specificity should be p_0
random_guess = np.random.binomial(n=1,p=p_1,size=len(ver_data))

# determine the correctness of the prediction from random guess
classification,sensitivity,specificity = cm.Check_correctness(random_guess,ver_data,plot=True,figsize=[6,4])


# %%
'''
Check with the prediction from the model
'''
# determine the correctness of the prediction from the model 
classification,sensitivity,specificity = cm.Check_correctness(prediction,ver_data,plot=True,figsize=[6,4])

pre,acc = cm.format(classification)


# %%

# %%
