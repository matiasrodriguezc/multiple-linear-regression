#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.linear_model import LinearRegression


#%%Load the Data
data = pd.read_csv('1.02. Multiple linear regression.csv')
data.head()
data.describe()


#%%Create the multiple Linear Regression
x = data[['SAT','Rand 1,2,3']]
y = data['GPA']


#%%Regression itself
reg = LinearRegression()
reg.fit(x,y)

#%%
reg.coef_


#%%
reg.intercept_


#%%Calculating the R-Squared
reg.score(x,y)


#%%Adjusted R-Squared Function
# There are different ways to solve this problem
# To make it as easy and interpretable as possible, we have preserved the original code
def adj_r2(x,y):
    r2 = reg.score(x,y)
    n = x.shape[0]
    p = x.shape[1]
    adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    return adjusted_r2
adj_r2(x,y)


#%%Feature Selection
# Import the feature selection module from sklearn
# This module allows us to select the most appopriate features for our regression
# There exist many different approaches to feature selection, however, we will use one of the simplest
from sklearn.feature_selection import f_regression
# We will look into: f_regression
# f_regression finds the F-statistics for the *simple* regressions created with each of the independent variables
# In our case, this would mean running a simple linear regression on GPA where SAT is the independent variable
# and a simple linear regression on GPA where Rand 1,2,3 is the indepdent variable
# The limitation of this approach is that it does not take into account the mutual effect of the two features
f_regression(x,y)
# There are two output arrays
# The first one contains the F-statistics for each of the regressions
# The second one contains the p-values of these F-statistics
#%%
# Since we are more interested in the latter (p-values), we can just take the second array
p_values = f_regression(x,y)[1]
p_values
#%%
# To be able to quickly evaluate them, we can round the result to 3 digits after the dot
p_values.round(3)


#%%Creating a Summary Table
# Let's create a new data frame with the names of the features
reg_summary = pd.DataFrame(data = x.columns.values, columns=['Features'])
reg_summary
# Then we create and fill a second column, called 'Coefficients' with the coefficients of the regression
reg_summary ['Coefficients'] = reg.coef_
# Finally, we add the p-values we just calculated
reg_summary ['p-values'] = p_values.round(3)
# Now we've got a pretty clean summary, which can help us make an informed decision about the inclusion of the variables 
reg_summary