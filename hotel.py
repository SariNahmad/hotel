import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print('data loaded')
df1 = pd.read_csv("C:/Users/Sari/Desktop/Landstar/hotel/data/Bodea - Choice based Revenue Management - Data Set - Hotel 1.csv")
#df2 = pd.read_csv("C:/Users/Sari/Desktop/Landstar/hotel/data/Bodea - Choice based Revenue Management - Data Set - Hotel 2.csv")
#df3 = pd.read_csv("C:/Users/Sari/Desktop/Landstar/hotel/data/Bodea - Choice based Revenue Management - Data Set - Hotel 3.csv")
#df4 = pd.read_csv("C:/Users/Sari/Desktop/Landstar/hotel/data/Bodea - Choice based Revenue Management - Data Set - Hotel 4.csv")
#df5 = pd.read_csv("C:/Users/Sari/Desktop/Landstar/hotel/data/Bodea - Choice based Revenue Management - Data Set - Hotel 5.csv")


print('concatenate data')
#df = pd.concat([df1, df2, df3, df4, df5])
df = df1

print(df.columns)
print(df.head())


print(df.Rate_Code.unique())