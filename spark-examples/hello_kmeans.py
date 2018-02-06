from numpy import array
from math import sqrt

from pyspark.mllib.clustering import KMeans, KMeansModel

def process_line(line):
	result = array([float(x) for x in line.split(' ')])
	return result

# Load and parse the data
data = sc.textFile("kmeans_data.txt")
parsedData = data.map(lambda x:process_line(x))


# Build the model (cluster the data)
clusters = KMeans.train(parsedData, 2, maxIterations=10, initializationMode="random")

# Evaluate clustering by computing Within Set Sum of Squared Errors
def error(point):
    center = clusters.centers[clusters.predict(point)]
    return sqrt(sum([x**2 for x in (point - center)]))

WSSSE = parsedData.map(lambda point: error(point)).reduce(lambda x, y: x + y)
print("Within Set Sum of Squared Error = " + str(WSSSE))

# Save and load model
clusters.save(sc, "target/org/apache/spark/PythonKMeansExample/KMeansModel")
sameModel = KMeansModel.load(sc, "target/org/apache/spark/PythonKMeansExample/KMeansModel")