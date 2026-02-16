import streamlit as st
import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("🎬 Movie Recommender System")

@st.cache_data
def load_data():
    movies = pd.read_csv("tmdb_5000_movies.csv")
    return movies

@st.cache_data
def preprocess(movies):
    movies = movies[['id','title','overview','genres','keywords']]
    movies.dropna(inplace=True)

    def convert(text):
        L = []
        for i in ast.literal_eval(text):
            L.append(i['name'])
        return L

    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)

    movies['tags'] = movies['overview'] + " " + movies['genres'].astype(str) + " " + movies['keywords'].astype(str)
    movies['tags'] = movies['tags'].apply(lambda x: x.lower())

    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(movies['tags']).toarray()

    similarity = cosine_similarity(vectors)

    return movies, similarity

movies = load_data()
movies, similarity = preprocess(movies)

movie_list = movies['title'].values
selected_movie = st.selectbox("Select a movie", movie_list)

def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = similarity[index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended = []
    for i in movies_list:
        recommended.append(movies.iloc[i[0]].title)
    return recommended

if st.button("Recommend"):
    recommendations = recommend(selected_movie)
    for movie in recommendations:
        st.write(movie)
