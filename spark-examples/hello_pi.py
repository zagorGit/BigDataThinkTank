
# coding: utf-8

# In[1]:


import findspark
findspark.init()


import pyspark
import random

sc = pyspark.SparkContext(appName="Pi")


# In[4]:


num_samples = 1000

def inside(p):     
  x, y = random.random(), random.random()
  return x*x + y*y < 1

def square(x):
    return x**2
    
local_list=list(range(0, num_samples))
print(local_list)


# In[6]:



parallel_list = sc.parallelize(local_list)

squares = parallel_list.map(square)
print(squares)
local_squares = squares.collect()
print(local_squares)


# In[10]:


tmp = parallel_list.filter(inside)
points = tmp.count()
print("Pi is roughly "+str(4.0 * points / num_samples))


# In[11]:



# In[8]:


from pyspark.ml.clustering import KMeans
from pyspark.sql import SQLContext

sqlContext = SQLContext(sc)

# Loads data.
dataset = sqlContext.read.format("libsvm").load("/Users/eftim/Repositories/BigDataThinkTank/spark-examples/data/sample_kmeans_data.txt")

# Trains a k-means model.
kmeans = KMeans().setK(2).setSeed(1)
model = kmeans.fit(dataset)

# Evaluate clustering by computing Within Set Sum of Squared Errors.
wssse = model.computeCost(dataset)
print("Within Set Sum of Squared Errors = " + str(wssse))


# In[12]:



# Shows the result.
centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)
    from numpy import array


# In[13]:



from pyspark.mllib.clustering import GaussianMixture, GaussianMixtureModel

def parse_line(line):
    result=array([float(x) for x in line.strip().split(' ')])
    return result
                  
# Load and parse the data
data = sc.textFile("/Users/eftim/Repositories/spark/data/mllib/gmm_data.txt")
parsedData = data.map(parse_line)
print(parsedData.count())


# In[14]:


print(parsedData.take(2))


# In[16]:



# Build the model (cluster the data)
gmm = GaussianMixture.train(parsedData, 2)

# Save and load model
gmm.save(sc, "/Users/eftim/Repositories/BigDataThinkTank/spark-examples/data/GaussianMixtureModel")


# In[17]:


sameModel = GaussianMixtureModel    .load(sc, "/Users/eftim/Repositories/BigDataThinkTank/spark-examples/data/GaussianMixtureModel")

# output parameters of model
for i in range(2):
    print("weight = ", gmm.weights[i], "mu = ", gmm.gaussians[i].mu,
          "sigma = ", gmm.gaussians[i].sigma.toArray())


# In[20]:


for x in parsedData.take(5):
    print(type(x), x)
    


# In[25]:


def flatten_list(x):
    i=0
    l= []
    for val in x:
        l.append((i,val))
        i+=1
    return l


# In[26]:


for x in parsedData.take(5):
    print(type(x), x, flatten_list(x))


# In[29]:


flat_data = parsedData.map(flatten_list)

for x in flat_data.take(5):
    print(x)


# In[35]:


vfd = parsedData.flatMap(flatten_list)

for x in vfd.take(10):
    print(x)



# In[36]:


print(vfd.count())


# In[37]:


gd = vfd.groupBy(lambda x: x[0])


# In[46]:


for x in gd.take(5):
    print('Group name:',x[0], 'Values in group:',len(x[1]))
    values=list(x[1])
    for y in values[:10]:
        print(y)
    


# In[47]:


def calc_stats(x):
    x2=[y[1] for y in x]
    return x2


# In[49]:


for x in gd.take(5):
    print('Group name:',x[0], 'Values in group:',calc_stats(x[1])[:10])


# In[50]:


import numpy as np
def calc_stats(x):
    x2=np.array([y[1] for y in x])
    stats = [np.mean(x2), np.std(x2), min(x2), max(x2)]
    return stats


# In[52]:


for x in gd.take(5):
    print('Group name:',x[0], 'Stats in group:')
    for st in calc_stats(x[1]):
        print(st)


# In[ ]:


|

