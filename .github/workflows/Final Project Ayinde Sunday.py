# -*- coding: utf-8 -*-
"""
Created on Mon Apr 02 03:05:01 2018

@author: AYINDE
"""
import scipy as stats
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib import style
from pandas.stats.moments import rolling_window 
import statsmodels.tsa.stattools as ts 
import pandas_datareader.data as web
import fix_yahoo_finance as yf
from sklearn.linear_model import LinearRegression
style.use('ggplot')
if __name__ == '__main__':
    #Import daily Gold Price from Yahoo finance 
    df= web.get_data_yahoo('GC=F', '1/4/2016','31/3/2018')
    # Extract date for close columns
    df2 = df['Close']
    # Drop rows with missing values
    df3 =df2.dropna()
    print(df3)
    # Plot the closing Price of GLD
    plt.plot(df3)
    plt.title('GOLD PRICE (OUNCE PER USD)')
    plt.xlabel('DATE')
    plt.ylabel('GOLD CLOSE PRICE')
    plt.show()
    # Defining explanatory variables
    df['S_2'] = df['Close'].shift(1).rolling(window=2).mean()
    df['S_8'] = df['Close'].shift(1).rolling(window=8).mean()
    df = df.dropna()
    fx = df[['S_2','S_8']]
    print(fx.head())
    # Defining dependent variables on explanatory variables
    y = df['Close']
    print(y.head())
    # Split the data into train and test dataset: 60% train dataset and 40% test dataset
    #fx_train and y_train are the training dataset
    #fx_test and y_test are the test data set
    t=.60
    t = int(t*len(df))
    # Train dataset
    fx_train =fx[:t]
    y_train = y[:t]
    # Test data set
    fx_test = fx[t:]
    y_test = y[t:]
    # Creating a linear regression
    linear = LinearRegression().fit(fx_train,y_train)
    print("GOLD PRICE =", round(linear.coef_[0],2),\
          "*2 Days Moving Average", round(linear.coef_[1],2),\
          "* 8 Days Moving Average+",round(linear.intercept_,2))
    # Predicting Gold price using Linear model created in the train dataset
    predicted_price = linear.predict(fx_test)
    predicted_price = pd.DataFrame(predicted_price,index=y_test.index,columns=['price'])
    predicted_price.plot(figsize=(10,5))
    y_test.plot()
    plt.title('GOLD PRICE EVOLUTION April 1,2016- Mar,31 2018')
    plt.legend(['predicted_price','actual-price'])
    plt.ylabel("GOLD PRICE")
    plt.show()
    r2_score = linear.score(fx[t:],y[t:])*100
    print(r2_score)
#    print(float("{0;.2f}".format(r2_score))
    #95.06947073406302
    
    

  
    
    