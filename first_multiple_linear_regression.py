
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn
seaborn.set()

#%%
# Load the data from a .csv in the same folder
data = pd.read_csv('1.02. Multiple linear regression.csv')

#%%
# Let's check what's inside this data frame
data

#%%
# This method gives us very nice descriptive statistics. We don't need this as of now, but will later on!
data.describe()

#%%Multiple Regression
# Following the regression equation, our dependent variable (y) is the GPA
y = data ['GPA']
# Similarly, our independent variable (x) is the SAT score
x1 = data [['SAT','Rand 1,2,3']]
# Add a constant. Esentially, we are adding a new column (equal in lenght to x), which consists only of 1s
x = sm.add_constant(x1)
# Fit the model, according to the OLS (ordinary least squares) method with a dependent variable y and an idependent x
results = sm.OLS(y,x).fit()
# Print a nice summary of the regression.
results.summary()