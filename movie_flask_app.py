import re, random, json
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


movies = pd.read_csv("movies.csv")  
ratings = pd.read_csv("ratings.csv")  

def clean_title(title):
    title = re.sub("[^a-zA-Z0-9 ]", "", title)
    return title

movies["clean_title"] = movies["title"].apply(clean_title)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
tfidf = vectorizer.fit_transform(movies["clean_title"])

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def search(title):
    title = clean_title(title)
    query_vec = vectorizer.transform([title])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    indices = np.argpartition(similarity, -5)[-5:]
    results = movies.iloc[indices].iloc[::-1]
    return results



def find_similar_movies(movie_id):
    similar_users = ratings[(ratings["movieId"] == movie_id) & (ratings["rating"] > 4)]["userId"].unique()
    similar_user_recs = ratings[(ratings["userId"].isin(similar_users)) & (ratings["rating"] > 4)]["movieId"]
    similar_user_recs = similar_user_recs.value_counts() / len(similar_users)

    similar_user_recs = similar_user_recs[similar_user_recs > .10]
    all_users = ratings[(ratings["movieId"].isin(similar_user_recs.index)) & (ratings["rating"] > 4)]
    all_user_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique())
    rec_percentages = pd.concat([similar_user_recs, all_user_recs], axis=1)
    rec_percentages.columns = ["similar", "all"]
    rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]
    rec_percentages = rec_percentages.sort_values("score", ascending=False)

    recommended_movie_ids = rec_percentages.head(10).index.tolist()
    recommended_movies = movies[movies["movieId"].isin(recommended_movie_ids)][["title", "genres"]]
    recommended_movies["score"] = rec_percentages.head(10)["score"].tolist()
    recommended_movies["movieId"] = recommended_movie_ids
    return recommended_movies


@app.route('/')
def home():
    return "Welcome to the Movie Recommendation API!"

@app.route('/random_movies', methods=['GET'])
def get_random_movies():
    with open('new_movies.json', 'r') as file:
        movie_data = json.load(file)

    random_movies = random.sample(movie_data, k=6)

    return jsonify(random_movies)

@app.route('/movies/<int:movie_id>', methods=['GET'])
def get_movie(movie_id):
    with open('new_movies.json', 'r') as file:
        movie_data = json.load(file)

    for movie in movie_data:
        if movie['movieId'] == movie_id:
            return jsonify(movie)

    return jsonify({'message': 'Movie not found'}), 404


@app.route('/search', methods=['GET'])
def search_movie():
    query = request.args.get('query')
    results = search(query)
    search_results = results.to_dict(orient='records')
    return jsonify(search_results)


@app.route('/recommendations', methods=['GET'])
def recommend_movies():
    query = request.args.get('query')
    results = search(query)
    movie_id = results.iloc[0]["movieId"]
    recommended_movies = find_similar_movies(movie_id)
    recommended_movies = recommended_movies.to_dict(orient='records')
    return jsonify(recommended_movies)

@app.route('/recommendations/<int:movie_id>', methods=['GET'])
def get_recommendations(movie_id):
    recommended_movies = find_similar_movies(movie_id)

    if recommended_movies.empty:
        return jsonify({'message': 'No recommendations found for the given movie ID.'}), 404

    recommended_movies = recommended_movies.to_dict(orient='records')
    return jsonify(recommended_movies)


if __name__ == '__main__':
    app.run(debug=True)












# import joblib
# import json

# model = joblib.load("model.joblib")
# with open("movies.json", 'r') as f:
#     movies = json.load(f)
  
# print(len(movies))
# predictions = model.predict(uid=0, iid =movies[0]["movieId"])  
# print(predictions)
# print(movies[0])