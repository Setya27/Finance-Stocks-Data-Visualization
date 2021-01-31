#!/usr/bin/env python
# coding: utf-8

# ## Autocomplete TAB
get_ipython().run_line_magic('config', 'Completer.use_jedi = False')


# Basic Stocks Data Analysis and Visualization
# 1. Import Dataset & Libraries
# 2. Perform EDA (Exploratory Data Analysis) and Basic Visualization
# 3. Perform Interactive Data Visualization
# 5. Calculate Stocks Daily Return
# 6. Calculate Correlation between Stocks Daily Return
# 7. Plot Histogram for Stocks Daily Return


# ## Import Dataset & Libraies
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from copy import copy
from scipy import stats
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

df = pd.read_csv('All Bank.csv')
stocks_df = df[809:].copy()
stocks_df = stocks_df.sort_values(by=['Date'])
stocks_df.set_index('Date', drop=True)


# ## PERFORM EXPLORATORY DATA ANALYSIS AND BASIC VISUALIZATION

# Check if data contains any null values
stocks_df.isnull().sum()

# Getting dataframe info
stocks_df.info()

# Define a function to plot the entire dataframe
def show_plot(df, fig_title):
    df.plot(x='Date', figsize=(15, 7), title=fig_title)
    plt.grid()
    plt.show()

# Plot the data
show_plot(stocks_df, 'STOCKS PRICES')


# ## PERFORM INTERACTIVE DATA VISUALIZATION

# Function to perform an interactive data plotting using plotly express
def interactive_plot (df, fig_title):
    fig = px.line(title=fig_title)
    
    # Loop through each stock (while ignoring time columns with index 0)
    for i in df.columns[1:]:
        fig.add_scatter(x=df['Date'], y=df[i], name=i)
        
    fig.show()

# Plot interactive chart
interactive_plot(stocks_df, 'STOCK PRICES INTERACTIVE PLOT')


# ## CALCULATE INDIVIDUAL STOCKS DAILY RETURNS

new_df = stocks_df.copy()

new_df = new_df.reset_index(drop=True)

# Let's define a function to calculate stocks daily returns (for all stocks) 
def daily_return(df):
    df_new = df.copy()
    
    # Loop through each stock (while ignoring time columns with index 0)
    for x in df.columns[1:]:
        
        # Loop through each row belonging to the stock
        for y in range(1, len(df)):
            
            # Calculate the percentage of change from the previous day
            df_new[x][y] = ((df[x][y] - df[x][y-1])/df[x][y-1])*100
        
        # set the value of first row to zero since the previous value is not available
        df_new[x][0] = 0
    return df_new

stocks_daily_return = daily_return(new_df)
my_stocks_daily_return = stocks_daily_return.set_index('Date')


# ## CALCULATE THE CORRELATIONS BETWEEN DAILY RETURNS
cm = stocks_daily_return.drop(columns=['Date']).corr()
plt.figure(figsize=(10, 10))
ax = plt.subplot()
sns.heatmap(cm, annot=True, ax=ax);


# ## PLOT THE HISTOGRAM FOR DAILY RETURNS
# Histogram of daily returns
# Stock returns are normally distributed with zero mean 
# Notice how Tesla Standard deviation is high indicating a more volatile stock
stocks_daily_return.hist(figsize=(10,10), bins=10);
df_hist = stocks_daily_return.copy()
df_hist = df_hist.drop(columns=['Date'])
data = []
for i in df_hist.columns:
    data.append(stocks_daily_return[i].values)
data
fig = ff.create_distplot(data, df_hist.columns)
fig.show()
