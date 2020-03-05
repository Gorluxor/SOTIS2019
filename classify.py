import glob
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from cluster import pretprocess

embedding = np.array(pd.read_csv("embedding.csv"))
centroids = pd.read_csv("centroids.csv")
print("a")

out = pretprocess("output//User 13_output_g.csv")

pca = PCA(n_components=2)
pca.fit(embedding)
X = pca.transform(embedding)

