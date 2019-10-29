#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 17:13:29 2019

@author: varakind
"""

import os
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)

#convert csv to dataframe
df = pd.read_csv("RUGChallenge2.csv")

#fit line to data set using statsmodels
Yall = df["weight"]
Xall = df["Time"]
Xall = sm.add_constant(Xall)
model = sm.OLS(endog=Yall,exog=Xall)
results = model.fit()

#print(r_value**2)
print ("Adjusted Rsquared for the data set is: " + str(results.rsquared_adj))

#need a list to store adjusted R2 from jackknifed samples
Radjusteds = []

#Outer loop creates empty lists and sets which observations to exclude
for i in range(len(df["Time"])):
    XTimes = []
    YWeights = []
    #inner loop populates list, excluding one data point, fits line, appends the R2adj
    for j in range(len(df["Time"])):
        if i != j:#the ith is excluded
            XTimes.append(df["Time"][j])
            YWeights.append(df["weight"][j])
    Y = YWeights
    X = XTimes
    X = sm.add_constant(X)
    JackedModel = sm.OLS(endog=Y,exog=X)
    JackedResults = JackedModel.fit()
    Radjusteds.append(JackedResults.rsquared_adj)
    
#ScatterPlot
plt.scatter(df["Time"],df["weight"],marker=r'$\heartsuit$')
plt.xlabel("Days Since Birth")
plt.ylabel("Weight (grams)")
plt.title("A Scatterplot")
plt.show()
            
#Make a historgram
plt.hist(Radjusteds)
plt.xlabel("Adjusted R2")
plt.ylabel("Frequency")
plt.title("Jackknifed R2Adj Distribution for weight ~ Time")
plt.show()

#make line/sequence graph
Radj = [results.rsquared_adj]*len(Radjusteds)
line1, = plt.plot(Radjusteds, label = "Jackknifed Sample R2Adj")
line2, = plt.plot(Radj, label = "R2Adj")
plt.xlabel("Index of Eliminated Data Point")
plt.ylabel("Adjusted R2")
plt.title("Jackknifed Samples")
plt.legend(handles=[line1, line2])
plt.show()

