import os
import requests
import json

def get_movie_image_url(title, year):
    url = "https://moviesdatabase.p.rapidapi.com/titles/search/title/{}".format(title)
    querystring = {"exact": "true", "year": str(year), "titleType": "movie"}
    headers = {
        "X-RapidAPI-Key": "d3ac844d16mshe80ce7a39d28ba8p1dbf89jsna3bacc69939d",
        "X-RapidAPI-Host": "moviesdatabase.p.rapidapi.com"
    }
    try:
        response = requests.get(url, headers=headers, params=querystring)
        data = response.json()
        if data is not None and "results" in data and data["results"]:
            print("getting image for" + title)
            return data["results"][0]["primaryImage"]["url"]
    except Exception as e:
        print("unable to get image for " + title, str(e))
    return ""


def process_movies(input_file, output_file):
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            new_movies = json.load(f)
            processed_movie_ids = [movie["movieId"] for movie in new_movies]
    else:
        new_movies = []
        processed_movie_ids = []

    with open(input_file, "r", encoding="utf-8") as f:
        movies = json.load(f)

    for movie in movies:
        movie_id = movie["movieId"]
        if movie_id in processed_movie_ids:
            continue  # Skip processing if movie has already been processed

        title = movie["title"]
        if "(" in title and ")" in title:
            year_start = title.rfind("(") + 1
            year_end = title.rfind(")")
            year = title[year_start:year_end]
            title = title[:year_start - 1].strip()
        else:
            year = ""
        image_url = get_movie_image_url(title, year)
        movie_data = {
            "movieId": movie_id,
            "title": title,
            "year": year,
            "image": image_url
        }
        new_movies.append(movie_data)
        processed_movie_ids.append(movie_id)

        # Save processed movies to the file incrementally
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(new_movies, f, ensure_ascii=False, indent=2)

    # Save the final processed movies to the file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(new_movies, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    input_file = "movies.json"
    output_file = "new_movies.json"
    process_movies(input_file, output_file)
