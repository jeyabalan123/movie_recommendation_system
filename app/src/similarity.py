from sklearn.metrics.pairwise import cosine_similarity

def compute_similarity(tfidf_matrix):
    """
    Computes cosine similarity between movie descriptions.
    """
    return cosine_similarity(tfidf_matrix)
