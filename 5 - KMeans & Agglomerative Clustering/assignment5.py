import numpy as np
import math

class K_MEANS:

	def __init__(self, k, t):
		#k_means state here
		#Feel free to add methods
		# t is max number of iterations
		# k is the number of clusters
		self.k = k
		self.t = t

		# list of sample indices for each cluster // 
		self.clusters = [[] for _ in range(self.k)]
		# storing mean feature vector for each cluster // actual samples
		self.Centroids = []

	def distance(self, centroids, datapoint):
		diffs = (centroids - datapoint)**2
		return np.sqrt(diffs.sum())

	def train(self, X):
		#training logic here
		#input is array of features (no labels)

		self.X = X
		self.nSamples, self.nFeatures = X.shape

		# Initializing centroids
		randomSampleIndexes = np.random.choice(self.nSamples, self.k, replace = False)
		self.Centroids = [self.X[idx] for idx in randomSampleIndexes]

		for _ in range(self.t):

			# update clusters
			self.clusters = self.createClusters(self.Centroids)

			# update centroids
			oldCentroids = self.Centroids
			self.Centroids = self.getCentroids(self.clusters) # this will assign the mean value of the clusters to the centroids
			

			# check if clusters have changed
			if self.checkIfConverged(oldCentroids, self.Centroids):
				break;

		return self.getClusterLabels(self.clusters)

		#return self.cluster
		#return array with cluster id corresponding to each item in dataset


	def getClusterLabels(self, clusters):
		labels = np.empty(self.nSamples)

		for clusterIndex, cluster in enumerate(clusters):
			for sampleIndex in cluster:
				labels[sampleIndex] = clusterIndex
		
		return labels

	def createClusters(self, Centroids):
		clusters = [[] for _ in range(self.k)]
		for idx, sample in enumerate(self.X):
			centroidIndex = self.closestCentroid(sample, Centroids)
			clusters[centroidIndex].append(idx)
		
		return clusters

	def closestCentroid(self, sample, Centroids):
		distances = [self.distance(sample, point) for point in Centroids]
		closestIndex = np.argmin(distances)

		return closestIndex

	def getCentroids(self, clusters):
		centroids = np.zeros((self.k, self.nFeatures))
		for clusterIndex, cluster in enumerate(clusters):
			clusterMean = np.mean(self.X[cluster], axis = 0)
			centroids[clusterIndex] = clusterMean
		return centroids

	def checkIfConverged(self, oldCentroids, Centroids):
		distances = [self.distance(oldCentroids[i], Centroids[i]) for i in range(self.k)]
		return sum(distances) == 0 # no more change in centroids


class AGNES:
	#Use single link method(distance between cluster a and b = distance between closest
	#members of clusters a and b
	def __init__(self, k):
		#agnes state here
		#Feel free to add methods
		# k is the number of clusters 
		self.k = k
		
	def distance(self, a, b):
		diffs = (a - b)**2
		return np.sqrt(diffs.sum())

	def train(self, X):
		clusters = [[i] for i in range(len(X))]

		distanceMatrix = np.zeros((len(X), len(X)))
		
		for i in range(len(X)):
			for j in range(len(X)):
				if(i!=j):
					distanceMatrix[i, j] = self.distance(X[i], X[j])
				else:
					distanceMatrix[i,j] = 0

		sortedDistances = sorted([(distanceMatrix[i, j], (i, j)) for i in range(len(X)) for j in range(i+1, len(X)) if i!=j], key = lambda x: x[0])

		while(len(clusters) != self.k and sortedDistances):

			nextPoint = sortedDistances.pop(0)
			idxPoint1 = nextPoint[1][0]
			idxPoint2 = nextPoint[1][1]

			nextGroupMerge = None

			for pointID in range(len(clusters)):

				cluster = clusters[pointID]

				if idxPoint1 in cluster and idxPoint2 in cluster:
					break

				elif idxPoint1 in cluster or idxPoint2 in cluster:
					if not nextGroupMerge:
						nextGroupMerge = pointID
					else:
						if pointID < nextGroupMerge:
							clusters[pointID].extend(clusters[nextGroupMerge])
							clusters.pop(nextGroupMerge)
						else:
							clusters[nextGroupMerge].extend(clusters[pointID])
							clusters.pop(pointID)
						break

		finalClusters = []

		for i in range(len(X)):
			for clusterID in range(len(clusters)):
				if i in clusters[clusterID]:
					finalClusters.append(clusterID)
					break

		self.cluster = np.array(finalClusters)
		#print(self.cluster)
		return self.cluster

	
	