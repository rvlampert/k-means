# K-Means

k-means clustering is a method of vector quantization, originally from signal processing, that aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean (cluster centers or cluster centroid), serving as a prototype of the cluster. This results in a partitioning of the data space into Voronoi cells. k-means clustering minimizes within-cluster variances (squared Euclidean distances), but not regular Euclidean distances, which would be the more difficult Weber problem: the mean optimizes squared errors, whereas only the geometric median minimizes Euclidean distances. For instance, better Euclidean solutions can be found using k-medians and k-medoids.

https://www.youtube.com/watch?v=4b5d3muPQmA

## Objective

Train a machine learning model capable of clustering a dataset, dividing it as best as possible

## How to run

Create a virtual environment, use ```make install-dependencies``` command to install the python dependencies, then use ```make run``` to run the program