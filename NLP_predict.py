import numpy as np
import pandas as pd
import sys
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

def read_data():
    wiki_movie_df = pd.read_json("wikidata-movies.json.gz", lines=True, encoding='UTF8')
    rating_df = pd.read_json("rotten-tomatoes.json.gz", lines=True, encoding='UTF8')
    genres_df = pd.read_json("omdb-data.json.gz", lines=True, encoding='UTF8')
    wiki_genres_df = pd.read_json("genres.json.gz", lines=True, encoding='UTF8')
    return wiki_movie_df, rating_df, genres_df, wiki_genres_df


def clean_data(genres_df):
    genres_df = genres_df.drop(columns=['omdb_awards', 'imdb_id'])
    genres_df['omdb_plot'] = genres_df['omdb_plot'].str.lower()
    genres_df.columns = ['genres', 'description']
    return genres_df



wiki_movie_df, rating_df, genres_df, wiki_genres_df = read_data()
genres_df = clean_data(genres_df)

genres_df = pd.concat([pd.Series(row['description'], row['genres'])
				for _, row in genres_df.iterrows()]).reset_index()
genres_df.columns = ['genres', 'desc']
genres_df.desc = genres_df.desc.str.split()

stop = stopwords.words('english')
for i in range(5):
	print(len(genres_df.iloc[i].desc))

genres_df.apply(lambda x: [item for item in x if item not in stop])
for i in range(5):
	print(len(genres_df.iloc[i].desc))

# count_vect = CountVectorizer()
# X_train_counts = count_vect.fit_transform(genres_df.description)



# print(genres_df.desc)

