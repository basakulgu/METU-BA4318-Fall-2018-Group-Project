#Cemal Atakan PARLAK - 2076552
#Başak ÜLGÜ - 2076966
#Muhammed Buğra TAŞIN - 2076628
#Şahin DERELİ - 2076156

import pandas as pd 	
import numpy as np
import math
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import statsmodels.api as sm

# dataframe
df = pd.read_csv("esenbogapax.csv", sep=';')
print(df.axes)
size = len(df)
train = df[0:size-50] 
test = df[size-50:size-1]

#functions
def summation (dataset):
    sum = 0
    for value in dataset:
        sum = sum + value
    return sum

def moving_average(train, test, value, windowsize ):
    # print("Moving Average")
    y_hat_avg = test.copy()
    y_hat_avg['Moving_Average'] = train[value].rolling(windowsize).mean().iloc[-1]
    rms1 = sqrt(mean_squared_error(test[value], y_hat_avg.Moving_Average))
    #print (rms)
    return rms1

def holt_winters(train, test, value, seasons):
    # print("Holt_Winter")
    y_hat_avg = test.copy()
    array = np.asarray(train[value])
    fit = ExponentialSmoothing( array ,seasonal_periods=seasons ,trend='add', seasonal='add',).fit()
    y_hat_avg['Holt_Winter'] = fit.forecast( len(test) )
    rms2 = sqrt(mean_squared_error(test[value], y_hat_avg.Holt_Winter))
    return rms2

errors = [0.0]
errors[0] = moving_average(train, test, value='PAX',windowsize=50)
sum_errors_moving = summation(errors)
print ("Error in Moving Average:", sum_errors_moving)

errors2 = [0.0]
errors2[0] = holt_winters(train, test, value='PAX',seasons=2)
sum_errors_holt_winters = summation(errors2)
print ("Error in Holt winters:", sum_errors_holt_winters)

print ("By checking the results we should choose the method which has the lowest error term:  Holt-Winters Method " )
print ("Please wait...")
    
train = df[0:size-1]
test = df [size-1:size-1]

totalpax= summation(df.PAX[-12:])
capacity= float(20000000)

while totalpax < capacity: 
    i = 0
    i = i+1
    y_hat_avg = test.copy()
    train = df[:]
    fit1 = ExponentialSmoothing(np.asarray(train['PAX']) ,seasonal_periods=2 ,trend='add', seasonal='add',).fit()
    y_hat_avg['Holt_Winter'] = fit1.forecast(i)
    forecastperiod = pd.DataFrame({
 
    "PAX": y_hat_avg['Holt_Winter']})
     
    df = pd.concat ([df, forecastperiod], ignore_index=True, sort=False)
    totalpax= summation(df.PAX[-12:])
  
else:
    totalmonths=(df.index[-1] - 130)
    print("Capacity is reached", totalmonths, " months later")
    