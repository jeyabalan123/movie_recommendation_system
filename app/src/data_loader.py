import pandas as pd
import numpy as np 
import ast
def load_data():
    """
    Loads and preprocesses the TMDB movie dataset.
    """
    df = pd.read_csv("dataset/tmdb_5000_movies.csv")
    
    # Select relevant columns
    df = df[['id', 'title', 'overview']]
    
    # Fill missing values
    df.fillna("", inplace=True)
    
    return df

def load_and_preprocess_data(filepath="data/tmdb_5000_movies.csv"):
    """Loads TMDB dataset and preprocesses it for clustering."""
    movies = pd.read_csv(filepath)

    # Extract genres from JSON-like strings
    movies["genre"] = movies["genres"].apply(lambda x: [g["name"] for g in ast.literal_eval(x)] if pd.notna(x) else [])

    # Convert genres to one-hot encoding
    unique_genres = sorted(set(g for genre_list in movies["genre"] for g in genre_list))
    genre_columns = pd.DataFrame([{g: 1 if g in genres else 0 for g in unique_genres} for genres in movies["genre"]])

    # Merge one-hot encoded genres with numerical features
    features = pd.concat([genre_columns, movies[["vote_average", "popularity", "release_date"]]], axis=1)

    # Normalize numerical features
    features = (features - features.mean()) / features.std()

    return movies, features