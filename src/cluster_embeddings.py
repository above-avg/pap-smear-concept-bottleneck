import numpy as np
from sklearn.cluster import KMeans


X = np.load("embeddings/patch_embeddings.npy")

print("Embedding matrix:", X.shape)


K = 30

kmeans = KMeans(n_clusters=K, random_state=0)
labels = kmeans.fit_predict(X)


np.save("clusters/cluster_labels.npy", labels)
np.save("clusters/cluster_centers.npy", kmeans.cluster_centers_)

print("Clustering finished")
print("Cluster labels shape:", labels.shape)