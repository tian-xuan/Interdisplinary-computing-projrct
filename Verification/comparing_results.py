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
# prediction = pd.readcsv('')


# %%
'''
Show the verification data
'''
# inport the verification data
ver_data = list(np.loadtxt('Verification_data.csv',dtype=int,delimiter=',',skiprows=1,usecols=(8)))

# determine the distribution of the outcomes of the verification data
num_of_0 = ver_data.count(0)
num_of_1 = ver_data.count(1)

p_0 = num_of_0/len(ver_data)
p_1 = num_of_1/len(ver_data)
print('The probability of true outcome is %.3f'%(p_1))
print('The probability of the false outcome is %.3f'%(p_0))

# plot the outcomes' distribution
fig1 = plt.figure(figsize=[6,4],tight_layout=True)
ax = plt.axes()
ax.bar([0,1],[num_of_0,num_of_1],0.6)
ax.set_xticks([0,1])
ax.set_xticklabels([0,1])
plt.xlabel('outcomes')
plt.ylabel('counts')
plt.show()

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
#classification,sensitivity,specificity = cm.Check_correctness(prediction,ver_data,plot=True,figsize=[6,4])


# %%

# %%
