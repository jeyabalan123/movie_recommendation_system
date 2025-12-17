from sklearn.cluster import KMeans

def apply_kmeans(tfidf_matrix, num_clusters=10):

    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(tfidf_matrix)
    
    return kmeans, cluster_labels
