# 🎬 Movie Recommender System

A content-based movie recommendation web app built with Python and Streamlit, powered by the TMDB 5000 dataset. Enter a movie you love and get instant personalized recommendations based on movie metadata like genres, cast, crew, and keywords.

---

## 🚀 Live Demo

👉 **[Try it here!](https://movie-recommender-system-4gpaucl6gdj3j4jlgfbmsx.streamlit.app/)**

---

## ✨ Features

- 🔍 Search any movie from the TMDB 5000 dataset
- 🎯 Content-based filtering using movie metadata
- 🖼️ Fetches movie posters via the TMDB API
- ⚡ Fast similarity computation using cosine similarity
- 🌐 Clean and interactive web UI via Streamlit

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Web App | Python, Streamlit |
| ML / NLP | scikit-learn, pandas, numpy |
| Dataset | TMDB 5000 Movies Dataset |
| API | TMDB API (for posters) |

---

## 📂 Project Structure

```
movie-recommender-system/
│
├── data/
│   ├── tmdb_5000_credits.csv    # TMDB credits dataset
│   └── requirements.txt         # Data-related dependencies
│
├── notebook/                    # Jupyter notebooks for EDA & model building
│
├── venv/                        # Virtual environment (not pushed to git)
│
├── app.py                       # Main Streamlit application
├── movie_list.pkl               # Preprocessed movie data
├── similarity.pkl               # Precomputed cosine similarity matrix
├── tmdb_5000_movies.csv         # TMDB movies dataset
├── requirements.txt             # Project dependencies
├── .gitignore
└── README.md
```

---

## ⚙️ How It Works

1. **Data Preprocessing** — Movie metadata (genres, keywords, cast, crew, overview) from the TMDB 5000 dataset is cleaned and merged into a single "tags" feature per movie.
2. **Vectorization** — Tags are converted into numerical vectors using `CountVectorizer`.
3. **Similarity Computation** — Cosine similarity is computed between all movie vectors and saved as `similarity.pkl`.
4. **Recommendation** — When a user picks a movie, the top 5 most similar movies are fetched from the similarity matrix and displayed with posters via the TMDB API.

---

## 🧑‍💻 Getting Started

### Prerequisites

- Python 3.8+
- A free [TMDB API Key](https://www.themoviedb.org/settings/api)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/krish0o7/movie-recommender-system.git
cd movie-recommender-system

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your TMDB API key in app.py
# Replace YOUR_API_KEY with your actual TMDB API key
```

### Run the App

```bash
streamlit run app.py
```

Then open your browser at `http://localhost:8501`

---

## 📦 Requirements

```
streamlit
pandas
numpy
scikit-learn
requests
```

> Install all with: `pip install -r requirements.txt`

---

## 📊 Dataset

This project uses the [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) from Kaggle, which contains metadata for 5,000 movies including:

- Movie overviews
- Genres
- Cast & Crew
- Keywords
- Ratings & Popularity

---

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙋‍♂️ Author

**Krish** — [@krish0o7](https://github.com/krish0o7)

---

> ⭐ If you found this project helpful, please give it a star!
