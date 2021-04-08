from sklearn.cluster import KMeans

def get_anchor(P, Q, n_cluster_u, n_cluster_i):
    u_estimator = KMeans(n_cluster_u)
    u_estimator.fit(P)
    u_clusterlabel = u_estimator.labels_
    u_centroids = u_estimator.cluster_centers_
    i_estimator = KMeans(n_cluster_i)
    i_estimator.fit(Q)
    i_clusterlabel = i_estimator.labels_
    i_centroids = i_estimator.cluster_centers_
    return u_clusterlabel, u_centroids, i_clusterlabel, i_centroids

