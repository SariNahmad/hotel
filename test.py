# session 1
##################
import pandas as pd
df = pd.read_csv('/Users/Sari/Desktop/sales_data/cluster_means_original.csv', sep =",")

df.append(dict(cluster=1, variable = 'a', value = 10), ignore_index = True)

df['n'] = 1
df['cluster'][df.index[df['n'] == 2]] = 23
df[df['cluster'] == 1]
df.sort('cluster', ascending = False)
df.sort(['cluster', 'value'], ascending = [False, True])
df[['cluster', 'value']].groupby(['cluster']).mean()
df[['cluster', 'value']].groupby(['cluster']).sum()
df.drop_duplicates()
df.append(dict(variable=1), ignore_index = True)

df.dropna() # drop any row with na 
df.dropna(subset=["cluster"]) #drop na's of certain column
df.fillna('x') # replace all nas
df.fillna(dict(cluster='x')) #replace column of nas
df.fillna(method = 'ffill') # pull down values to fill, bfill to backfill


# new data frame
newdf = pd.DataFrame(dict(cluster=[1,2,3,6], color=['r','d','s','t']))
df.merge(newdf, how='left', on=['cluster']) # inner, outer, left
#left_on and right_on rather than just “on”
#left_index and right_index rather than just “on”

df2 = pd.DataFrame(dict(foo=['one','one','one','two','two','two'], 
	bar=['a','b','c','a','b','c'], baz=[1,2,3,4,5,6]))
df2.pivot(index='foo', columns='bar', values='baz')

##################
df = pd.read_csv('/Users/Sari/Desktop/sales_data/cluster_means_original.csv', sep =",")
df.pivot(index='variable', columns='cluster', values='value')
df2 = df.pivot(index='cluster', columns='variable', values='value')


df3 = pd.DataFrame({'A': {0: 'a', 1: 'b', 2: 'c'},'B': {0: 1, 1: 3, 2: 5}, 'C': {0: 2, 1: 4, 2: 6}, 'D': {0: 6, 1: 3, 2: 9}})
pd.melt(df3, id_vars=['A','B'], value_vars=['C','D'])

# session 2: numpy 
##################
import numpy as np
ar1 = np.array([1, 2, 3])
ar1.shape

ar1 = np.array([[1,2,3],[1]]) # shape 2L

ar = np.array([[1,2,3],[1,2,3]]) #(2L, 3L)
ar[0]
ar[0,1]
ar[[0,1]]
ar[[0,1],1] # rows 0, 1, column 1 
ar[[0,1],::-1] 
ar[1:, 1:]

np.ones(5)
np.ones([3,2])
np.ones(ar.shape)

np.random.rand(2,5)
np.random.randint(2,5,[10,10])

ar.flatten()

ar.mean()
ar.mean(axis=1)
ar.std()
ar.max()
ar.min()


#1. numpy.linalg — fantastic module for more advanced linear algebra needs (matrix decomposition, eigen{values,vectors}, etc.)
#2. numpy.dtype — can be customized and used to make NumPy arrays feel more like R/Pandas data frames
#3. numpy.random — tons of statistical distributions; great for simulation or automating data transformation

# session 3: matplotlib
##################
import matplotlib.pyplot as plt
plt.plot([1,2,3,4,5], [1,4,9,16,25])
plt.axis([0,10,0,50])
plt.show()

plt.plot([1,2,3,4,5], [1,4,9,16,25], 'o')

plt.plot([1,2,3,4,5], [1,4,9,16,25], 'r--', linewidth=3.0)

#...functions previously defined
x = np.arange(-5, 5, 0.01)
plt.subplot(211) #numrows, numcol, fignum
plt.plot(x, f(x), 'bo')
plt.subplot(212)
plt.plot(x, g(x), 'r--')
plt.show()

plt.plot(x, f(x))
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Graph with Label, Title, and Text!')
plt.text(0, 3, 'Hello Opex!', ha='center')
plt.show()


plt.plot(x[x>0], x[x>0]**3)
plt.yscale('log')
plt.show()


x = 100 + 15*np.random.randn(100000)
plt.hist(x, 50) # histogram with 50 bins
plt.grid(True) # turn grid on
