import requests
import streamlit as sl
import pandas as pd
import ast
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

no_of_features = 50
no_of_recommendations = 5


movies = pd.read_csv('movie names.csv')
crew = pd.read_csv('crew names.csv')
crew.rename(columns={'movie_id': 'id'}, inplace=True)
movies = movies.merge(crew, on="id")
movies.drop(
    columns={"budget", "homepage", "overview", "popularity", "revenue", "runtime", "status", "tagline", "title_x",
             "title_y", "vote_average", "vote_count", "spoken_languages"}, inplace=True)
movies.dropna(inplace=True)
movies.isnull().sum()


def fetch_posters(id_no):
    response = requests.get(
        "https://api.themoviedb.org/3/movie/{}?api_key=e0035eec8b740ca0eb379c23708822a4".format(id_no)).json()
    return "https://image.tmdb.org/t/p/w500/" + response['poster_path']


def getDetails(obj):
    data = []
    for i in ast.literal_eval(obj):
        data.append(i["name"])
    return data


def getDirector(obj):
    director = []
    for i in ast.literal_eval(obj):
        if i["job"] == "Director":
            director.append(i["name"])
    return director


def getProducer(obj):
    producer = []
    for i in ast.literal_eval(obj):
        if i["job"] == "Producer":
            producer.append(i["name"])
    return producer


def getEditor(obj):
    editor = []
    for i in ast.literal_eval(obj):
        if i["job"] == "Editor":
            editor.append(i["name"])
    return editor


def stemer(s):
    ps = PorterStemmer()
    temp_lst = []
    for i in s.split():
        temp_lst.append(ps.stem(i))
    return temp_lst


movies.loc[:, "genres"] = movies.loc[:, "genres"].apply(getDetails)
movies.loc[:, "keywords"] = movies.loc[:, "keywords"].apply(getDetails)
movies.loc[:, "production_companies"] = movies.loc[:, "production_companies"].apply(getDetails)
movies.loc[:, "production_countries"] = movies.loc[:, "production_countries"].apply(getDetails)
movies.loc[:, "cast"] = movies.loc[:, "cast"].apply(getDetails)
movies.loc[:, "directors"] = movies.loc[:, "crew"].apply(getDirector)
movies.loc[:, "producers"] = movies.loc[:, "crew"].apply(getProducer)
movies.loc[:, "editors"] = movies.loc[:, "crew"].apply(getEditor)
movies.drop(columns="crew", inplace=True)
movies.loc[:, "genres"] = movies.loc[:, "genres"].apply(lambda x: [i.replace(" ", "") for i in x])
movies.loc[:, "keywords"] = movies.loc[:, "keywords"].apply(lambda x: [i.replace(" ", "") for i in x])
movies.loc[:, "production_companies"] = movies.loc[:, "production_companies"].apply(
    lambda x: [i.replace(" ", "") for i in x])
movies.loc[:, "production_countries"] = movies.loc[:, "production_countries"].apply(
    lambda x: [i.replace(" ", "") for i in x])
movies.loc[:, "cast"] = movies.loc[:, "cast"].apply(lambda x: [i.replace(" ", "") for i in x])
movies.loc[:, "directors"] = movies.loc[:, "directors"].apply(lambda x: [i.replace(" ", "") for i in x])
movies.loc[:, "producers"] = movies.loc[:, "producers"].apply(lambda x: [i.replace(" ", "") for i in x])
movies.loc[:, "editors"] = movies.loc[:, "editors"].apply(lambda x: [i.replace(" ", "") for i in x])
movies.loc[:, "release_date"] = movies.loc[:, "release_date"].apply(lambda x: [str(int(x[0:4]) // 100)])
movies.loc[:, 'tags'] = movies.loc[:, "genres"] + movies.loc[:, "keywords"] + movies.loc[:,
                                                                              "production_companies"] + movies.loc[:,
                                                                                                        "production_countries"] + movies.loc[
                                                                                                                                  :,
                                                                                                                                  "release_date"] + movies.loc[
                                                                                                                                                    :,
                                                                                                                                                    "cast"] + movies.loc[
                                                                                                                                                              :,
                                                                                                                                                              "directors"] + movies.loc[
                                                                                                                                                                             :,
                                                                                                                                                                             "producers"] + movies.loc[
                                                                                                                                                                                            :,
                                                                                                                                                                                            "editors"]
training_df = movies.loc[:, ["id", "original_title", "tags"]]
training_df.loc[:, "tags"] = training_df.loc[:, "tags"].apply(lambda x: " ".join(x))
training_df.loc[:, 'tags'] = training_df.loc[:, 'tags'].apply(lambda x: x.lower())
training_df.loc[:, "tags"] = training_df.loc[:, "tags"].apply(stemer).apply(lambda x: " ".join(x))

vectorizer = CountVectorizer(max_features=no_of_features, stop_words='english')
vectors = vectorizer.fit_transform(training_df["tags"])
similarity_matrix = cosine_similarity(vectors)


def get_movies(movie):
    movie_index = [i for i in training_df[training_df["original_title"] == movie].index]
    similarity = similarity_matrix[[i for i in movie_index][0]]
    movie_dict = {}
    for i in ((sorted(list(enumerate(similarity)), key=lambda x: x[1], reverse=True))[1:no_of_recommendations + 1]):
        movie_dict.update({(training_df[training_df.index == i[0]]["id"].values[0]): (
            training_df[training_df.index == i[0]]["original_title"].values[0])})
    return movie_dict


sl.title("WHAT TO WATCH NOW")
sl.subheader("GET YOUR FAV MOVIE HERE")

option = sl.selectbox(
    "Write down the movie name"
    , training_df["original_title"])
recommendations=get_movies(option).items()

if sl.button('Get Recommendations'):
    for i in recommendations:
        sl.write(i[1])
        sl.image(fetch_posters(i[0]), width=333)
