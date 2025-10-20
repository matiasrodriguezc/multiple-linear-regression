#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.linear_model import LinearRegression


#%%Load Data
data = pd.read_csv('real_estate_price_size_year.csv')
data.head()
data.describe()


#%%Create the Regression
### Declare the dependent and the independent variables
x = data[['size','year']]
y = data['price']
###Scale the inputs
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x)
x_scaled = scaler.transform(x)
### Regression
reg = LinearRegression()
reg.fit(x_scaled,y)


#%%Find the intercept
reg.intercept_


#%%Find the coefficients
reg.coef_


#%%Calculate the R-Squared
reg.score(x,y)


#%%Calculate the Adjusted R-Squared
# Let's use the handy function we created
def adj_r2(x,y):
    r2 = reg.score(x,y)
    n = x.shape[0]
    p = x.shape[1]
    adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    return adjusted_r2
adj_r2(x,y)


#%%Making predictions
new_data = [[750,2009]]
new_data_scaled = scaler.transform(new_data)
reg.predict(new_data_scaled)

#%%Calculate the univariate p-values of the variables
from sklearn.feature_selection import f_regression
f_regression(x_scaled,y)
p_values = f_regression(x,y)[1]
p_values
p_values.round(3)

#%%Create a summary table with our findings
reg_summary = pd.DataFrame(data = x.columns.values, columns=['Features'])
reg_summary ['Coefficients'] = reg.coef_
reg_summary ['p-values'] = p_values.round(3)
reg_summary