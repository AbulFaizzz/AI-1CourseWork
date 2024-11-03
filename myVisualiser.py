import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def cluster_and_visualise(datafilename, K, featurenames):

    rows = np.genfromtxt(datafilename, delimiter=',')
    num_features = rows.shape[1]

    kmeans = KMeans(K, n_init=10)
    kmeans.fit(rows)
    labels = kmeans.predict(rows)
    
    fig, axs = plt.subplots(num_features, num_features, figsize=(15, 15), facecolor="#fff1db")
    plt.set_cmap('summer')
    
    for x_axis in range(len(rows[0])):
        for y_axis in range(len(rows[0])):
            graph = axs[x_axis, y_axis]

            if x_axis != y_axis:
                graph.scatter(rows[:, x_axis], rows[:, y_axis], c=labels)
                graph.set_xlabel(featurenames[x_axis])
                graph.set_ylabel(featurenames[y_axis])

            else:
                data1 = rows[:, x_axis][labels == 0]
                data2 = rows[:, x_axis][labels == 1]
                graph.hist([data1, data2], bins=10, color=['red', 'green'], label=['Cluster 0', 'Cluster 1'])
                graph.legend()
                graph.set_xlabel(featurenames[x_axis])
                graph.set_ylabel("frequency")

    fig.suptitle(f"Abul Faiz S2102901")
    fig.tight_layout()
    plt.savefig("myVisualisation.jpg")

    return fig, axs