import glob
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

files = glob.glob("output/*_g.csv")
print("Number of files to process: " + str(files.__len__()))
def convert_time_series_to_series(info):
   return "".join(info[' s'] + info[' s'] * info[' fpogd'])

def split(word):
    return [char for char in word]


from sgt import Sgt


#print(files)
if __name__ == "__main__":
    all_seq = []
    for f in files:
        list_of_sequences = []
        time_data = pd.read_csv(f, usecols=['Seq', ' fpogx', ' fpogy', ' fpogd', ' s'])
        time_data[' s'] = time_data[' s'].astype('str')
        for i in range(1,26):
            selected = time_data.loc[time_data['Seq'] == i]
            if isinstance(selected, pd.DataFrame):
                list_of_sequences.append(convert_time_series_to_series(selected))
        sequences = [split(x) for x in list_of_sequences]
        all_seq.extend([x for x in sequences if x.__len__() > 0])
    print(all_seq[0])
    sgt = Sgt(kappa=10, lengthsensitive=False)
    embedding = sgt.fit_transform(corpus=all_seq)

    pca = PCA(n_components=2)
    pca.fit(embedding)
    X = pca.transform(embedding)
    #print(np.sum(pca.explained_variance_ratio_))

    def get_cmap(n, name='hsv'):
        '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
        RGB color; the keyword argument name must be a standard mpl colormap name.'''
        return plt.cm.get_cmap(name, n)


    df = pd.DataFrame(X)
    num_clusters = 8
    kmeans = KMeans(n_clusters=num_clusters, max_iter=300)
    kmeans.fit(df)
    labels = kmeans.predict(df)
    centroids = kmeans.cluster_centers_
    fig = plt.figure(figsize=(5, 5))
    #colmap = {1: 'r', 2: 'g', 3: 'b', }
    cmap = get_cmap(num_clusters)
    colors = list(map(lambda x: cmap(x+1), labels))
    plt.scatter(df[0], df[1], color=colors, alpha=0.5, edgecolor=colors)
    plt.show()

    # # Elbow Method of Optimal K
    # SSD = []
    # K = range(1, 15)
    # for k in K:
    #     km = KMeans(n_clusters=k, max_iter=300)
    #     km = km.fit(df)
    #     SSD.append(km.inertia_)
    #
    # plt.plot(K, SSD, 'bx-')
    # plt.xlabel('k')
    # plt.ylabel('Sum of squared distances')
    # plt.title('Elbow Method of Optimal k ')
    # plt.show()
