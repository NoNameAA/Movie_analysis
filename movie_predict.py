import warnings
warnings.filterwarnings("ignore")
import process_data
import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
import matplotlib.pyplot as plt
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier




def predict_by_all(df, X_list):
	X = df[X_list]
	y = df['profit']
	

	# Normalize each features
	scaler = StandardScaler()
	scaler.fit(X)
	X = scaler.transform(X)


	# Data split
	x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.5)

	#SVM Classifier
	svc_model = SVC(kernel='linear', C=21)
	svc_model.fit(x_train, y_train)
	prediction = svc_model.predict(x_test)
	SVC_score = svc_model.score(x_test, y_test)

	#Logistic Regression
	LR_model = LogisticRegression()
	LR_model.fit(x_train, y_train)
	LR_score = LR_model.score(x_test, y_test)

	NB_model = GaussianNB()
	NB_model.fit(x_train, y_train)
	NB_score = NB_model.score(x_test, y_test)

	return SVC_score, LR_score, NB_score


def feature_selection(df, X_list):
	corr = df[X_list].corr()
	sb.heatmap(corr, cmap="Blues", annot=True)
	plt.show()

	X = df[X_list].copy()
	scaler = StandardScaler()
	scaler.fit(X)
	X = scaler.transform(X)
	y = df['profit']


	forest = RandomForestClassifier()
	forest.fit(X, y)
	importances = forest.feature_importances_
	print("The importance of features are: (sorted)")
	result = []
	for k, v in sorted(zip(map(lambda x: round(x, 4), importances), X_list), reverse=True):
		print(v + ': ' + str(k))
		result.append((v, k))

	selected_list = []
	n = 2
	for i in range(n):
		selected_list.append(result[i][0])

	selected_list.append('profit')
	selected_df = df[selected_list]
	selected_df = selected_df.reset_index(drop=True)	

	return selected_df, selected_list

def feature_reduction(df, X_list):
	X = df[X_list]	
	scaler = StandardScaler()
	scaler.fit(X)
	X = scaler.transform(X)

	pca = PCA(2)
	pca.fit(X)
	X = pca.transform(X)

	index = ['x1', 'x2']
	pca_df = pd.DataFrame(X, columns=index)
	df = df.reset_index()
	pca_df['profit'] = df['profit']

	return pca_df

def predict_by_2x(df, normalized):
	X = df.iloc[:, 0:2]
	y = df.iloc[:, -1]
	if normalized == False:		
		scaler = StandardScaler()
		scaler.fit(X)
		X = scaler.transform(X)
		

	# Data split
	x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.5)


	#SVM Classifier
	svc_model = SVC(kernel='linear', C=21)
	svc_model.fit(x_train, y_train)
	SVC_score = svc_model.score(x_test, y_test)

	#Logistic Regression
	LR_model = LogisticRegression()
	LR_model.fit(x_train, y_train)
	LR_score = LR_model.score(x_test, y_test)

	NB_model = GaussianNB()
	NB_model.fit(x_train, y_train)
	NB_score = NB_model.score(x_test, y_test)


	return SVC_score, LR_score, NB_score

def show_distribution(df):	
	X = df.iloc[:, 0:2]	
	scaler = StandardScaler()
	scaler.fit(X)
	X = scaler.transform(X)
	temp_df = pd.DataFrame(data=X, columns=['x', 'y'])
	temp_df['profit'] = pd.Series(df['profit'], index=temp_df.index)
	temp_df['x'] = temp_df[temp_df['x'] < 3] # remove some outliers
	sb.lmplot(x='x', y='y', fit_reg=False, data=temp_df, hue='profit', \
			markers=["x", "o"], palette={True:'b', False:'r'})
	plt.show()


def predict_profit(wiki_movie_df, rating_df):
	wiki_movie_df = wiki_movie_df.set_index('imdb_id').join(rating_df.set_index('imdb_id'), how='left', lsuffix='wiki_movie_df')


	wiki_movie_df = wiki_movie_df.reset_index()

	info_list = ['audience_ratings',
				'audience_percent', 
				'audience_average', 
				'critic_percent', 
				'critic_average',
				'made_profit']	

	wiki_movie = wiki_movie_df[info_list].copy().reset_index()

	wiki_movie = wiki_movie[(wiki_movie.made_profit == 1.0) | (wiki_movie.made_profit == 0.0) ]

	wiki_movie['profit'] = pd.Series(wiki_movie.made_profit == 1.0, index=wiki_movie.index)
	wiki_movie = wiki_movie.drop(columns=['made_profit'])	
	wiki_movie = wiki_movie[wiki_movie.isnull().sum(axis=1) == 0]


	X_list = ['audience_ratings',
				'audience_percent', 
				'audience_average', 
				'critic_percent', 
				'critic_average']

	SVC_score_all, LR_score_all, NB_score_all = predict_by_all(wiki_movie, X_list)

	fs_df, selected_list = feature_selection(wiki_movie, X_list)

	# print(fs_df)

	pca_df = feature_reduction(wiki_movie, X_list)

	SVC_score_pca, LR_score_pca, NB_score_pca = predict_by_2x(pca_df, True)

	SVC_score_fs, LR_score_fs, NB_score_fs= predict_by_2x(fs_df, False)

	score_list = [
					SVC_score_all, LR_score_all, NB_score_all, \
					SVC_score_pca, LR_score_pca, NB_score_pca, \
					SVC_score_fs, LR_score_fs, NB_score_fs
	]

	method_list = [
					"The accuracy of model with all rating features by SVM Classifier",
					"The accuracy of model with all rating features by Logistic Regression",
					"The accuracy of model with all rating features by Naive bayes Classifier",
					"The accuracy of model with PCA transformed features by SVM Classifier",
					"The accuracy of model with PCA transformed features by Logistic Regression",
					"The accuracy of model with PCA transformed features by Naive Bayes Classifier",
					"The accuracy of model with Top-2 important features by SVM Classifier",
					"The accuracy of model with Top-2 important features by Logistic Regression",
					"The accuracy of model with Top-2 important features by Naive Bayes Classifier",
	]

	for k, v in sorted(zip(map(lambda x: round(x, 4), score_list), method_list), reverse=True):
		print(v + ': ' + str(k)) 	

	sb.set()
	show_distribution(fs_df)


if __name__ == "__main__":
	wiki_movie_df, rating_df, genres_df, wiki_genres_df = process_data.read_data()
	predict_profit(wiki_movie_df, rating_df)	
	# print(wiki_movie_df)
	