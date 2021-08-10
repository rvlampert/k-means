import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Generating sample chart for example

X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
plt.scatter(X[:, 0], X[:, 1], s=50, edgecolor='black')
plt.figure()

# Training the algorithm

model = KMeans( n_clusters=4,
                n_init=10, max_iter=300, 
                tol=1e-04, random_state=0
)
groups = model.fit_predict(X)

# Plotting the graph

plt.scatter(X[groups == 0, 0], X[groups == 0, 1],
            s=50, c='lightgreen', edgecolor='black',
            label='cluster 1'
)
plt.scatter(X[groups == 1, 0], X[groups == 1, 1],
            s=50, c='orange', edgecolor='black',
            label='cluster 2'
)
plt.scatter(X[groups == 2, 0], X[groups == 2, 1],
            s=50, c='lightblue', edgecolor='black',
            label='cluster 3'
)
plt.scatter(
          X[groups == 3, 0], X[groups == 3, 1],
          s=50, c='green', edgecolor='black',
          label='cluster 4'
)
plt.scatter(
    model.cluster_centers_[:, 0], model.cluster_centers_[:, 1],
    s=150, marker='*',
    c='red', edgecolor='black',
    label='centroids'
)
plt.legend(scatterpoints=1)
plt.grid()
plt.show()