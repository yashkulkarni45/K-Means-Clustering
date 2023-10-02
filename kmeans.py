import random
import numpy as np

class Kmeans:
    def __init__(self, n_clusters = 2, max_iter = 100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None

    def fit_predict (self, x):
        random_index = random.sample(range(0,x.shape[0]),self.n_clusters)
        self.centroids = x[random_index]

        for i in range(self.max_iter):
           #assign clusters
            cluster_group = self.assignClusters(x) 
            old_centroids = self.centroids
            #move centroids 
            self.centroids = self.move_centroids(x,cluster_group)
            #check if finished or not
            if(old_centroids == self.centroids).all():
                break

        return cluster_group 

       
    def assignClusters(self,x):
        cluster_group = []
        distances = []

        for row in x:
            for centroid in self.centroids:
                distances.append(np.sqrt(np.dot(row-centroid,row-centroid)))
            min_dist = min(distances)
            index_pos = distances.index(min_dist)
            cluster_group.append(index_pos)
            distances.clear()

        return np.array(cluster_group)

    def move_centroids(self,x,cluster_groups):
        new_centroids = []

        cluster_type = np.unique(cluster_groups)

        for type in cluster_type:
            new_centroids.append(x[cluster_groups == type].mean(axis=0)) #0 for column
        return np.array(new_centroids)