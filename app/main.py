import streamlit as st
import pandas as pd
from src.data_loader import load_data
from src.feature_extraction import extract_features
from src.similarity import compute_similarity
from src.clustering import apply_kmeans
from src.recommender import recommend_movies

# Load and preprocess data
df = load_data()
tfidf, tfidf_matrix = extract_features(df)
similarity_matrix = compute_similarity(tfidf_matrix)

# Apply K-Means Clustering
num_clusters = 10  # You can adjust the number of clusters
kmeans, cluster_labels = apply_kmeans(tfidf_matrix, num_clusters)

# Add cluster labels to the dataset
df["cluster"] = cluster_labels

# Streamlit UI
st.title("🎬 Movie Recommendation System with K-Means Clustering")

# User Input
movie_title = st.text_input("Enter a movie title:")

if st.button("Get Recommendations"):
    recommendations = recommend_movies(movie_title, df, similarity_matrix, cluster_labels)
    
    if "Movie not found!" in recommendations:
        st.error("Movie not found! Please enter a valid title.")
    else:
        st.success("Here are some recommended movies from the same cluster:")
        for movie in recommendations:
            st.write(f"✅ {movie}")
