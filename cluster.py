import glob
import pandas as pd
import numpy as np
from matplotlib import gridspec
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, OPTICS, SpectralClustering
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances_argmin_min, silhouette_score
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

files = glob.glob("output/*_g.csv")
test = files[3]
print("Excluding user:" + str(test))
files.remove(test)
print("Number of files to process: " + str(files.__len__()))
def convert_time_series_to_series(info):
   return "".join(info[' s'] + info[' s'] * info[' fpogd'])

def split(word):
    return [char for char in word]


from sgt import Sgt

def pretprocess(filename):
    list_of_sequences = []
    time_data = pd.read_csv(filename, usecols=['Seq', ' fpogx', ' fpogy', ' fpogd', ' s'])
    time_data[' s'] = time_data[' s'].astype('str')
    for i in range(1, 26):
        selected = time_data.loc[time_data['Seq'] == i]
        if isinstance(selected, pd.DataFrame):
            list_of_sequences.append(convert_time_series_to_series(selected))
    sequences = [split(x) for x in list_of_sequences]
    return [x for x in sequences if x.__len__() > 0]
#print(files)



if __name__ == "__main__":
    all_seq = []
    for f in files:
        # list_of_sequences = []
        # time_data = pd.read_csv(f, usecols=['Seq', ' fpogx', ' fpogy', ' fpogd', ' s'])
        # time_data[' s'] = time_data[' s'].astype('str')
        # for i in range(1,26):
        #     selected = time_data.loc[time_data['Seq'] == i]
        #     if isinstance(selected, pd.DataFrame):
        #         list_of_sequences.append(convert_time_series_to_series(selected))
        # sequences = [split(x) for x in list_of_sequences]
        all_seq.extend(pretprocess(f))
    print(all_seq[0])

    sgt = Sgt(kappa=10, lengthsensitive=False)
    embedding = sgt.fit_transform(corpus=all_seq)

    pd.DataFrame(embedding).to_csv(path_or_buf='embedding.csv', index=False)
    pd.DataFrame(embedding).head()
    import umap
    #pca = PCA(n_components=2)
    #pca.fit(embedding)
    #X = pca.transform(embedding)

    reducer = umap.UMAP()
    trained_umap = reducer.fit(embedding)
    X = trained_umap.transform(embedding)

    #print(np.sum(pca.explained_variance_ratio_))

    def get_cmap(n, name='plasma'):
        '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
        RGB color; the keyword argument name must be a standard mpl colormap name.'''
        return plt.cm.get_cmap(name, n)


    # df = pd.DataFrame(X)
    # num_clusters = 7
    # kmeans = KMeans(n_clusters=num_clusters, max_iter=300)
    # kmeans.fit(df)
    # labels = kmeans.predict(df)
    # centroids = kmeans.cluster_centers_
    # pd.DataFrame(centroids).to_csv(path_or_buf="centroids.csv")
    # fig = plt.figure(figsize=(5, 5))
    # #colmap = {1: 'r', 2: 'g', 3: 'b', }
    # cmap = get_cmap(num_clusters)
    # colors = list(map(lambda x: cmap(x+1), labels))
    # plt.scatter(df[0], df[1], color=colors, alpha=0.5, edgecolor=colors)
    # plt.show()
    #
    # cluster_labels = kmeans.fit_predict(df)
    # silhouette_avg = silhouette_score(X, cluster_labels)
    # print("The average silhouette_score is :", silhouette_avg)

    Q = []
    #for i in range(7, 8):
    df = pd.DataFrame(X)
    #df =  pd.DataFrame(StandardScaler().fit_transform(df))
    num_clusters = 7
    clustering = SpectralClustering(n_clusters=num_clusters, assign_labels="discretize", random_state=0)
    clustering.fit(df)
    labels = clustering.labels_
    # fig = plt.figure(figsize=(5, 5))
    # colmap = {1: 'r', 2: 'g', 3: 'b', }
    cmap = get_cmap(num_clusters)
    colors = list(map(lambda x: cmap(x + 1), labels))
    plt.scatter(df[0], df[1], color=colors, alpha=0.5, edgecolor=colors)
    plt.show()

    cluster_labels = clustering.fit_predict(df)
    silhouette_avg = silhouette_score(X, clustering.labels_)


    #Q.append([i, silhouette_avg])
    print("The average silhouette_score is :", silhouette_avg)
    # plt.plot([x for x, y in Q], [y for x, y in Q])
    # plt.title("Number of cluster compared to sillouete score")
    # plt.show()



    # # # Metoda za određivanja epsilona
    # neigh = NearestNeighbors(n_neighbors=2)
    # nbrs = neigh.fit(df)
    # distances, indices = nbrs.kneighbors(df)
    # distances = np.sort(distances, axis=0)
    # distances = distances[:, 1]
    # distances *= 10
    # from kneed import DataGenerator, KneeLocator
    # x_vel = range(0, len(distances))
    # kneedle = KneeLocator(x_vel, distances, S=3.0, curve='convex', direction='increasing', interp_method='polynomial')
    # print(round(kneedle.knee, 3))
    # print(distances[round(kneedle.knee)])
    # plt.plot(distances)
    # plt.show()

    # T1 = []
    # T2 = []
    # T3 = []
    # for e in np.arange(0.07, 0.47, 0.001):
    #     data = StandardScaler().fit_transform(df)
    #     db = DBSCAN(eps=e, min_samples=3).fit(data)
    #     labels = db.labels_
    #     n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    #     n_noise_ = list(labels).count(-1)
    #     #print('Estimated number of clusters: %d, %f' % (n_clusters_, e))
    #     #print('Estimated number of noise points: %d, %f' % (n_noise_, e))
    #     T1.append([e, n_clusters_])
    #     T2.append([e, n_noise_])
    #     T3.append([e, metrics.silhouette_score(X, labels)])
    #     #print("Silhouette Coefficient: %0.3f"
    #     #      % metrics.silhouette_score(X, labels))
    # plt.figure(figsize=(10, 7))
    # G = gridspec.GridSpec(1, 3)
    # ax1 = plt.subplot(G[0, 0])
    # ax2 = plt.subplot(G[0, 1])
    # ax3 = plt.subplot(G[0, 2])
    #
    # ax1.plot([x for x, y in T1], [y for x, y in T1])
    # ax2.plot([x for x, y in T2], [y for x, y in T2])
    # ax3.plot([x for x, y in T3], [y for x, y in T3])
    #
    # ax1.title.set_text("Number of clusters")
    # ax2.title.set_text("Number of noise")
    # ax3.title.set_text("Silhouette score")
    # plt.tight_layout()
    # plt.show()

    # db!!
    # data = StandardScaler().fit_transform(df)
    # db = DBSCAN(eps=0.367, min_samples=3).fit(data)
    # labels = db.labels_
    # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # print("Num of clusters %d" % n_clusters_ )
    # n_noise_ = list(labels).count(-1)
    # #print('Estimated number of clusters: %d, %f' % (n_clusters_, e))
    # #print('Estimated number of noise points: %d, %f' % (n_noise_, e))
    #
    # cluster_labels = db.fit_predict(data)
    # silhouette_avg = silhouette_score(X, cluster_labels)
    # print("The average silhouette_score is :", silhouette_avg)
    # cmap = get_cmap(n_clusters_, "tab20")
    # colors = list(map(lambda x: cmap(x+1), labels))
    # import matplotlib.patches as mpatches
    #
    # plt.scatter(data[:, 0], data[:, 1], c=cluster_labels, cmap=plt.cm.Paired)
    # plt.scatter(data[])


    # T = []
    # for e in cmap._lut:
    #     if e[0] == 0 and e[1] == 0 and e[2] == 0 and e[3] == 0:
    #         break
    #     red_patch = mpatches.Patch(color=e, label='Cluter ' + str(k))
    #     T.append(red_patch)
    #     k = k + 1
    # plt.legend(handles=T)
    #plt.show()


    #test
    # out = pretprocess("output//User 13_output_g.csv")
    #
    # sgt2 = Sgt(kappa=10, lengthsensitive=False)
    # embedding2 = sgt2.fit_transform(corpus=out)
    # pca = PCA(n_components=2)
    # pca.fit(embedding2)
    # X = pca.transform(embedding2)
    # df2 = pd.DataFrame(X)
    # cmap2 = get_cmap(1)
    # colors2 = list(map(lambda x: cmap(x + 1), labels))
    # plt.scatter(df2[0], df2[1])
    # plt.show()
   # print(centroids)
    #print(X)
    #closest, _ = pairwise_distances_argmin_min(centroids, X)

    #print(closest)
    #
    # Elbow Method of Optimal K
    # SSD = []
    # K = range(1, 15)
    # for k in K:
    #     km = KMeans(n_clusters=k, max_iter=300)
    #     km = km.fit(df)
    #     SSD.append(km.inertia_)
    # font = {'family': 'normal',
    #         'weight': 'bold',
    #         'size': 22}
    # plt.plot(K, SSD, 'bx-')
    # plt.xlabel('k')
    # plt.ylabel('Sum of squared distances')
    # plt.title('Elbow Method of Optimal k ')
    # plt.show()
    # # kneedle = KneeLocator(K, SSD, S=2.0, curve='convex', direction='decreasing')
    # # print(round(kneedle.knee, 3))
    # # print(distances[round(kneedle.knee)])
    # from yellowbrick.cluster import KElbowVisualizer
    # model = KMeans()
    # visualizer = KElbowVisualizer(
    #     model, k=(4, 12), metric='calinski_harabasz', timings=False, font=font
    # )
    #
    # visualizer.fit(df)  # Fit the data to the visualizer
    # visualizer.show()

    # # data show
    # plt.scatter(df[0], df[1])
    # plt.xlabel("Feature 0")
    # plt.ylabel("Feature 1")
    # plt.show()



    # Elbow Method of Optimal K !@#!#!@##
    # clustering = DBSCAN(eps=3, min_samples=2).fit(X)
    # SSD = []
    # K = range(1, 15)
    # for k in K:
    #     #km = DBSCAN(eps=3, min_samples=2)
    #     km = OPTICS(min_samples=2).fit(df)
    #     #SSD.append(km.inertia_)
    #     SSD.append()
    #
    # plt.plot(K, SSD, 'bx-')
    # plt.xlabel('k')
    # plt.ylabel('Sum of squared distances')
    # plt.title('Elbow Method of Optimal k ')
    # plt.show()



    #     clust = OPTICS(min_samples=2).fit(df)
    #
    #     space = np.arange(len(X))
    #     reachability = clust.reachability_[clust.ordering_]
    #     labels = clust.labels_[clust.ordering_]
    #
    #     plt.figure(figsize=(10, 7))
    #     G = gridspec.GridSpec(2, 3)
    #     ax1 = plt.subplot(G[0, :])
    #
    #     colors = ['g.', 'r.', 'b.', 'y.', 'c.']
    #     for klass, color in zip(range(0, 5), colors):
    #         Xk = space[labels == klass]
    #         Rk = reachability[labels == klass]
    #         ax1.plot(Xk, Rk, color, alpha=0.3)
    #     ax1.plot(space[labels == -1], reachability[labels == -1], 'k.', alpha=0.3)
    #     ax1.plot(space, np.full_like(space, 2., dtype=float), 'k-', alpha=0.5)
    #     ax1.plot(space, np.full_like(space, 0.5, dtype=float), 'k-.', alpha=0.5)
    #     ax1.set_ylabel('Reachability (epsilon distance)')
    #     ax1.set_title('Reachability Plot')
    #
    #     plt.tight_layout()
    #     plt.show()

