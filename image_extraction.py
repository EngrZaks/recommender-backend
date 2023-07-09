import json
import requests

# Load movies from the JSON file
# with open('movies.json') as file:
#     movies = json.load(file)

movies = [
  {
    "movieId": 1,
    "title": "Toy Story (1995)",
    "genres": "Adventure|Animation|Children|Comedy|Fantasy"
  },
  {
    "movieId": 2,
    "title": "Jumanji (1995)",
    "genres": "Adventure|Children|Fantasy"
  },
  {
    "movieId": 3,
    "title": "Grumpier Old Men (1995)",
    "genres": "Comedy|Romance"
  },
  {
    "movieId": 4,
    "title": "Waiting to Exhale (1995)",
    "genres": "Comedy|Drama|Romance"
  },
  {
    "movieId": 5,
    "title": "Father of the Bride Part II (1995)",
    "genres": "Comedy"
  },
  {
    "movieId": 6,
    "title": "Heat (1995)",
    "genres": "Action|Crime|Thriller"
  }]

# Initialize the list for new movie data
new_movies = []

# Iterate over the movies
for movie in movies:
    # Extract movie name (without year) and year
    title = movie['title']
    year_start = title.rfind('(') + 1
    year_end = title.rfind(')')
    year = title[year_start:year_end]
    movie_name = title[:year_start - 1].strip()

    # Make request to the movie API
    url = "https://moviesdatabase.p.rapidapi.com/titles/search/title/{}".format(movie_name)
    querystring = {
        "exact": "true",
        "year": year,
        "titleType": "movie"
    }
    headers = {
        "X-RapidAPI-Key": "d3ac844d16mshe80ce7a39d28ba8p1dbf89jsna3bacc69939d",
        "X-RapidAPI-Host": "moviesdatabase.p.rapidapi.com"
    }
    response = requests.get(url, headers=headers, params=querystring)
    response_json = response.json()

    # Extract image URL from the response
    if 'results' in response_json and response_json['results']:
        primary_image = response_json['results'][0].get('primaryImage', {})
        image_url = primary_image.get('url', None)
    else:
        image_url = None

    # Create new movie data with image URL
    new_movie = {
        "movieId": movie['movieId'],
        "title": movie_name,
        "year": year,
        "image": image_url
    }

    # Append new movie to the list
    new_movies.append(new_movie)

# Save new movie data to a JSON file
with open('new_movies.json', 'w') as file:
    json.dump(new_movies, file, indent=2)
