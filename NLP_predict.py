import numpy as np
import pandas as pd
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.naive_bayes import MultinomialNB
from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB
import string
import re

def read_data():
    # wiki_movie_df = pd.read_json("wikidata-movies.json.gz", lines=True, encoding='UTF8')
    # rating_df = pd.read_json("rotten-tomatoes.json.gz", lines=True, encoding='UTF8')
    genres_df = pd.read_json("omdb-data.json.gz", lines=True, encoding='UTF8')
    # wiki_genres_df = pd.read_json("genres.json.gz", lines=True, encoding='UTF8')
    # return wiki_movie_df, rating_df, genres_df, wiki_genres_df
    return genres_df


def clean_data(genres_df):
    genres_df = genres_df.drop(columns=['omdb_awards', 'imdb_id'])
    genres_df['omdb_plot'] = genres_df['omdb_plot'].str.lower()

    # pattern = r'[^\w\s]'

    # genres_df['omdb_plot'] = genres_df['omdb_plot'].apply( \
    #                     lambda x: re.sub(pattern, '', x))

    # genres_df.columns = ['genres', 'description']
    # genres_df = pd.concat([pd.Series(row['description'], row['genres'])
    #                 for _, row in genres_df.iterrows()]).reset_index()


    genres_df.columns = ['genres', 'desc']

    # genres_df['genres'] = genres_df['genres'].apply(lambda x: '.'.join(x))

    # print(genres_df.genres)

    # genres_df.desc = genres_df.desc.str.split()

    # stop = stopwords.words('english')

    # genres_df['desc'] = genres_df['desc'].apply( \
    #                     lambda x: [item for item in x if item not in stop])

    # genres_df['desc'] = genres_df['desc'].apply( \
    #                     lambda x: ' '.join(y for y in x))

    return genres_df.iloc[:100]


# wiki_movie_df, rating_df, genres_df, wiki_genres_df = read_data()
genres_df = read_data()

genres_df = clean_data(genres_df)

x_train, x_test, y_train, y_test = model_selection.train_test_split ( \
                                genres_df.desc, genres_df.genres, test_size=0.1)

# transformer model
count_vect = CountVectorizer()
x_train_counts = count_vect.fit_transform(x_train)
# print(tt)

encoder = LabelEncoder()
encoder.fit(genres_df.genres)
y_train = encoder.transform(y_train)
# print(y_train[:10])

# # tfidf transform
tfidf_transformer = TfidfTransformer(use_idf=True)
x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

clf = MultinomialNB().fit(x_train_tfidf, y_train)

# # test the performance of classifier
x_test_counts = count_vect.transform(x_test)
x_test_tfidf = tfidf_transformer.transform(x_test_counts)
y_test = encoder.transform(y_test)
score = clf.score(x_test_tfidf, y_test)
y_predict = clf.predict(x_test_tfidf)
prob = clf.predict_proba(x_test_tfidf)
sort_prob = np.argsort(prob)

tx = range(14)
# print(tx)
ll = encoder.inverse_transform(tx)
for i in tx:
    print(i, ll[i],)

# print(prob.round(2))
# print('arg_sort:')
print(sort_prob)

print("predict:")
print(y_predict)

print("observe:")
print(y_test)

print("label predict:")
print(encoder.inverse_transform(y_predict))

print('label observe:')
print(encoder.inverse_transform(y_test))
# print(score)


# clf_nb = GaussianNB()
# # print(y_train.shape)
# clf_nb.fit(x_train_tfidf.toarray(), y_train)
# nb_score = clf_nb.score(x_test_tfidf.toarray(), y_test)
# print(nb_score)
# print(X_train_counts)

# print(genres_df.desc)

