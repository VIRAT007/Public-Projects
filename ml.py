import pandas as pd
import ast
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv('movie names.csv')
crew = pd.read_csv('crew names.csv')
movies.shape

crew.rename(columns={'movie_id': 'id'}, inplace=True)
movies.shape

movies = movies.merge(crew, on="id")
# movies.head(5)

movies.drop(
    columns={"budget", "homepage", "overview", "popularity", "revenue", "runtime", "status", "tagline", "title_x",
             "title_y", "vote_average", "vote_count", "spoken_languages"}, inplace=True)

movies.dropna(inplace=True)
movies.isnull().sum()


def getDetails(obj):
    data = []
    for i in ast.literal_eval(obj):
        data.append(i["name"])
    return data


movies["genres"] = movies["genres"].apply(getDetails)
movies["keywords"] = movies["keywords"].apply(getDetails)
movies["production_companies"] = movies["production_companies"].apply(getDetails)
movies["production_countries"] = movies["production_countries"].apply(getDetails)
movies["cast"] = movies["cast"].apply(getDetails)


def getDirector(obj):
    director = []
    for i in ast.literal_eval(obj):
        if (i["job"] == "Director"):
            director.append(i["name"])
    return director


def getProducer(obj):
    producer = []
    for i in ast.literal_eval(obj):
        if (i["job"] == "Producer"):
            producer.append(i["name"])
    return producer


def getEditor(obj):
    editor = []
    for i in ast.literal_eval(obj):
        if (i["job"] == "Editor"):
            editor.append(i["name"])
    return editor


def stemer(s):
    ps = PorterStemmer()
    temp_lst = []
    for i in s.split():
        temp_lst.append(ps.stem(i))
    return (temp_lst)


movies["directors"] = movies["crew"].apply(getDirector)
movies["producers"] = movies["crew"].apply(getProducer)
movies["editors"] = movies["crew"].apply(getEditor)
movies.drop(columns="crew", inplace=True)
movies["genres"] = movies["genres"].apply(lambda x: [i.replace(" ", "") for i in x])
movies["keywords"] = movies["keywords"].apply(lambda x: [i.replace(" ", "") for i in x])
movies["production_companies"] = movies["production_companies"].apply(lambda x: [i.replace(" ", "") for i in x])
movies["production_countries"] = movies["production_countries"].apply(lambda x: [i.replace(" ", "") for i in x])
movies["cast"] = movies["cast"].apply(lambda x: [i.replace(" ", "") for i in x])
movies["directors"] = movies["directors"].apply(lambda x: [i.replace(" ", "") for i in x])
movies["producers"] = movies["producers"].apply(lambda x: [i.replace(" ", "") for i in x])
movies["editors"] = movies["editors"].apply(lambda x: [i.replace(" ", "") for i in x])
movies["release_date"] = movies["release_date"].apply(lambda x: [str(int(x[0:4]) // 100)])
movies['tags'] = movies["genres"] + movies["keywords"] + movies["production_companies"] + movies[
    "production_countries"] + movies["release_date"] + movies["cast"] + movies["directors"] + movies["producers"] + \
                 movies["editors"]
training_df = movies[["id", "original_title", "tags"]]
training_df["tags"] = training_df["tags"].apply(lambda x: " ".join(x))
training_df['tags'] = training_df['tags'].apply(lambda x: x.lower())
training_df["tags"] = training_df["tags"].apply(stemer).apply(lambda x: " ".join(x))


def get_movies(movie):
    vectorizer = CountVectorizer(max_features=5000, stop_words='english')
    vectors = vectorizer.fit_transform(training_df["tags"])
    similarity_matrix = cosine_similarity(vectors)
    movie_index = [i for i in training_df[training_df["original_title"] == movie].index]
    similarity = similarity_matrix[[i for i in movie_index][0]]
    movie_dict = {}
    for i in ((sorted(list(enumerate(similarity)), key=lambda x: x[1], reverse=True))[1:5]):
        movie_dict.update({(training_df[training_df.index == i[0]]["id"].values[0]): (
            training_df[training_df.index == i[0]]["original_title"].values[0])})
    return movie_dict


print(get_movies("Batman"))
