import assignment5 as models
import numpy as np
import sys

if(sys.version_info[0] < 3):
    raise Exception("This assignment must be completed using Python 3")

def load_data(path):
    data = np.genfromtxt(path, delimiter=',', dtype=float)
    return data[:,:-1], data[:,-1].astype(int)

X, y = load_data("county_statistics.csv")

#Initialization
#k_means
k = 3
t=50 #max iterations
k_means = models.K_MEANS(k, t)

#AGNES
k = 3
agnes = models.AGNES(k)

#Train
k_means.train(X)
agnes.train(X)


from sklearn import metrics

print(metrics.silhouette_score(X, k_means.train(X), metric='euclidean'))
print(metrics.silhouette_score(X, agnes.train(X), metric='euclidean'))
