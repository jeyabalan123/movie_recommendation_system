from sklearn.feature_extraction.text import TfidfVectorizer

def extract_features(df):
    """
    Converts movie overviews into numerical vectors using TF-IDF.
    """
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["overview"])
    
    return tfidf, tfidf_matrix
