import numpy as np
import pandas as pd
import sys


def read_data():

	wiki_movie_df = pd.read_json("wikidata-movies.json.gz", lines=True, encoding='UTF8')

	rating_df = pd.read_json("rotten-tomatoes.json.gz", lines=True, encoding='UTF8')
	# rating_df['na'] = pd.Series(rating_df.isnull().sum(axis=1) >= 3, index=rating_df.index)		
	# rating_df = rating_df.drop(rating_df[rating_df.na == True].index)
	
	#Rating_df 0 missing value -- 16732
	#Rating_df 2 missing value -- 15749
	#Rating_df 3 missing value -- 7
	#Rating_df >= 4 missing value -- 7742
	#Rating.shape -- 40230

	genres_df = pd.read_json("omdb-data.json.gz", lines=True, encoding='UTF8')
	# omdb_awards None -- 2810
	# genres_df.shape -- [9676, 5]

	wiki_genres_df = pd.read_json("genres.json.gz", lines=True, encoding='UTF8')
	# print(wiki_movie_df[wiki_movie_df['made_profit'].isnull() == False]['made_profit'])

	# print(rating_df)
	# test = wiki_movie_df.set_index('imdb_id').join(rating_df.set_index('imdb_id'), how='left', lsuffix='wiki_movie_df')

	# print(rating_df.shape)
	# print(wiki_movie_df.shape)
	# print(test['critic_percent'].isnull().sum())
	# audience_average 32000 not null
	# audience_percent 32000 not null
	# audience_rating 39000 not null
	# critic_average 17000 not null
	# critic_percent 17000 not null

	# rating_df['na'] = pd.Series(rating_df.isnull().sum(axis=1) >= 4, index=rating_df.index)	
	# genres_df['na'] = pd.Series(genres_df['omdb_awards'] == "N/A", index=genres_df.index)
	# wiki_movie_df = wiki_movie_df.sort_values(by=['wikidata_id'])
	# wiki_genres_df = wiki_genres_df.sort_values(by=['wikidata_id'])

	# wiki_genres_df.set_index('wikidata_id')
	# wiki_movie_df.set_index('wikidata_id')

	########## Join wiki_movie_df with wiki_genres ###########
	# wiki_movie_df = wiki_movie_df.set_index('wikidata_id').join(wiki_genres_df.set_index('wikidata_id'), how='left')
	# joined_wiki_movie = pd.Series(test['genre_label'].isnull(), index=test.index)
	# print(test['na'].sum())
	###########################################################

	# print(wiki_genres_df)
	# print(wiki_movie_df)

	return wiki_movie_df, rating_df, genres_df, wiki_genres_df



if __name__ == "__main__":
	# test function here
	read_data()

