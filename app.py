import streamlit as st
import pandas as pd
import numpy as np
import ast
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", layout="wide")

st.title("🎬 Movie Recommender System")
st.markdown("### Discover Movies Similar to Your Favorites")
st.markdown("---")

# -------------------------------
# Load Data
# -------------------------------
@st.cache_data
def load_data():
    movies = pd.read_csv("tmdb_5000_movies.csv")
    return movies

# -------------------------------
# Preprocess Data
# -------------------------------
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

# -------------------------------
# Fetch Poster from TMDB
# -------------------------------
def fetch_poster(movie_id):
    api_key = "279736120d4205ac9b8d85989f2a6826"   # 🔥 PUT YOUR TMDB API KEY HERE
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}"
    data = requests.get(url).json()
    poster_path = data.get('poster_path')

    if poster_path:
        return "https://image.tmdb.org/t/p/w500/" + poster_path
    else:
        return "https://via.placeholder.com/500x750?text=No+Image"

# -------------------------------
# Recommend Function
# -------------------------------
def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = similarity[index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    recommended_posters = []

    for i in movies_list:
        movie_id = movies.iloc[i[0]].id
        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_posters.append(fetch_poster(movie_id))

    return recommended_movies, recommended_posters

# -------------------------------
# Main App Logic
# -------------------------------
movies = load_data()
movies, similarity = preprocess(movies)

movie_list = movies['title'].values
selected_movie = st.selectbox("Select a movie", movie_list)

if st.button("Recommend"):
    with st.spinner("Finding best movies for you..."):
        names, posters = recommend(selected_movie)

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.text(names[0])
            st.image(posters[0])

        with col2:
            st.text(names[1])
            st.image(posters[1])

        with col3:
            st.text(names[2])
            st.image(posters[2])

        with col4:
            st.text(names[3])
            st.image(posters[3])

        with col5:
            st.text(names[4])
            st.image(posters[4])
