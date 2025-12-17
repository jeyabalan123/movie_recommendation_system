def recommend_movies(title, df, similarity_matrix, cluster_labels):
    """
    Recommends movies based on content similarity within the same cluster.
    """
    if title not in df["title"].values:
        return ["Movie not found! Please check the title."]
    
    # Get the index of the input movie
    movie_index = df[df["title"] == title].index[0]
    
    # Find the cluster of the input movie
    movie_cluster = cluster_labels[movie_index]

    # Get movies in the same cluster
    cluster_movies = df[cluster_labels == movie_cluster].index.tolist()

    # Compute similarity only within the cluster
    similarity_scores = [(i, similarity_matrix[movie_index][i]) for i in cluster_movies if i != movie_index]

    # Sort by similarity score and get the top 5 recommendations
    sorted_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[:5]

    recommended_movies = [df.iloc[i[0]]["title"] for i in sorted_movies]

    return recommended_movies

