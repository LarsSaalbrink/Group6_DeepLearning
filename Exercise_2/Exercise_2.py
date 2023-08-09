#%%
import pandas as pd
import numpy as np
import matplotlib as mpl
import statsmodels.api as sm
import statsmodels.graphics.api as smg
import matplotlib.pyplot as plt

#%%
##### Part 1 #####
df = pd.read_csv('auto.csv')

# Remove the rows in the ‘horsepower’ column that has the value ‘?’
df_clean = df[df.horsepower != '?']
#Select all but the name column and convert them to numeric
df_int = df_clean.copy().drop(columns='name').apply(pd.to_numeric)

#%%
##### Part 5 #####
#Column names:
#mpg, cylinders, displacement, horsepower, weight, acceleration, model_year, origin, name

#Preprocess the data to reduce span of numbers, ensuring that they are of similar magnitude
df_int[['cylinders','displacement','horsepower','weight','acceleration','model_year']] = np.log(df_int[['cylinders','displacement','horsepower','weight','acceleration','model_year']])

#%%
##### Part 2 #####
names = list(df_clean.columns)
for i in range(0,8):
    fig, ax = plt.subplots()  # Create a figure containing a single axes.
    ax.set_ylabel(names[i])
    ax.scatter(df_int[names[0]],df_int[names[i]]);  # Plot some data on the axes.

plt.show()

#%%
##### Part 3 & 4 #####
Y = df_int['mpg']
X = df_int[['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year']]
X = sm.add_constant(X)

mod = sm.OLS(Y,X)
results = mod.fit()

print(results.params)
print(results.summary())
print(results.cov_params())
fig = smg.plot_partregress_grid(results) #, exog_idx=x_names)
fig.tight_layout(pad=1.0)

fig.show()
#Block script from exiting so picture stays open
input("Press enter to exit")

# %%
