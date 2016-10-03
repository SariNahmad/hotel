import pandas as pd
import numpy as np

df1 = pd.read_csv("C:/Users/Sari/Desktop/Landstar/hotel/data/Bodea - Choice based Revenue Management - Data Set - Hotel 1.csv")
df2 = pd.read_csv("C:/Users/Sari/Desktop/Landstar/hotel/data/Bodea - Choice based Revenue Management - Data Set - Hotel 2.csv")
df3 = pd.read_csv("C:/Users/Sari/Desktop/Landstar/hotel/data/Bodea - Choice based Revenue Management - Data Set - Hotel 3.csv")
df4 = pd.read_csv("C:/Users/Sari/Desktop/Landstar/hotel/data/Bodea - Choice based Revenue Management - Data Set - Hotel 4.csv")
df5 = pd.read_csv("C:/Users/Sari/Desktop/Landstar/hotel/data/Bodea - Choice based Revenue Management - Data Set - Hotel 5.csv")


df1.head()
df1.tail(2)
df1.index
df1.columns
df1.values
df1.describe()
df1.shape
df1.T
df1.sort_index(axis=1, ascending=False)
#df1.sort_values(by='Booking_Date')
df1['Booking_Date']
df1.Booking_Date
df1[0:3]
df1.loc[:,['Hotel_ID','Booking_ID']]
df1.loc[0:3,['Hotel_ID','Booking_ID']]
df1.loc[0,['Hotel_ID','Booking_ID']] # reduce dimensions
df1.loc[0,['Hotel_ID']]
df1.loc[0,'Hotel_ID'] # returns scalar
df1.at[0, 'Hotel_ID'] # returns scalar
df1.iloc[3] # by position
df1.iloc[3:5]
df1.iloc[3:5, 0:2]
df1.iloc[1,1]
df1.iat[1,1]
df1[df1.Booking_ID < 3]
df1['new_column'] = 1 # add new column
df1.head(2)
df1[df1['Booking_ID'].isin([1,3,5])]
df1.at[0:3,'Booking_ID'] = 100 # change values
df1.head(5)
df1.iat[0,0] = 0
df1.loc[0,'Hotel_ID']
df1.dropna(how='any')
df1.fillna(value=555)
pd.isnull(df1)
df1.mean(1)
# df1.apply(np.cumsum)

df1['Distribution_Channel'].str.lower().head()

pd.concat([df1[1:3], df1[4:6]]) #c concatenate
pd.concat([df1, df2, df3]).head()
pd.concat([df1, df2, df3]).tail()

#pd.merge(left, right, on='key')


# df.append(s, ignore_index=True)


df1.groupby('Hotel_ID').sum()

#####################################

df = pd.concat([df1, df2, df3, df4, df5])

df.groupby('Hotel_ID').sum()
df.groupby('Hotel_ID').mean()

#pd.pivot_table(df, values = 'Number_of_Rooms', index = ['Hotel_ID'], columns = ['Check_In_Date'])


df.columns
df["test"] = df["Distribution_Channel"].astype("category")
#df["test"].cat.categories = ["very good", "good", "very bad"]

import matplotlib.pyplot as plt
ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
ts = ts.cumsum()
ts
ts.plot()
x = pd.DataFrame(np.random.randn(1000, 4), index=ts.index, columns=['A', 'B', 'C', 'D'])

df = df.cumsum()
plt.figure(); df.plot(); plt.legend(loc='best')