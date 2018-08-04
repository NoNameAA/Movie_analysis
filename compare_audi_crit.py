import process_data
import pandas as pd
import numpy as np
import sys
from scipy import stats
import matplotlib.pyplot as plt
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import seaborn



def get_data(rating_df):
	rating_df = rating_df.dropna()
	#rating_df = rating_df[rating_df['audience_ratings'] >= 8000]
	rating_df = rating_df.sort_values(by=['audience_ratings'], ascending = False)
	rating_df = rating_df[:300]
	#print(rating_df['audience_ratings'])
	#print(rating_df)
	audience_average = rating_df['audience_average']
	critic_average = rating_df['critic_average']
	audience_percent = rating_df['audience_percent']
	critic_percent = rating_df['critic_percent']
	#
	# Because the audience average is out of 5, critic average is out of 10 and both audience percent
	# and critic percent are out of 100, the stardant of them are different
	critic_average = critic_average*10
	audience_average = audience_average*20
	# print(audience_average)
	# print(critic_average)
	print("----- Data -----")
	# print(rating_df.count())
	# print("\n")
	print(audience_average.head())
	print(critic_average.head())
	print(audience_percent.head())
	print(critic_percent.head())
	print("\n")

	return audience_average, critic_average, audience_percent, critic_percent


def do_anova(audience_average, critic_average, audience_percent, critic_percent):
	anova = stats.f_oneway(audience_average, critic_average, audience_percent, critic_percent)
	print("----- Anova -----")
	print("pvalue of anova: ", anova.pvalue)
	if (anova.pvalue < 0.05):
		print("Because of pvalue < 0.05, there is a difference between the means of the groups.")
	return anova.pvalue

def do_post_hoc(audience_average, critic_average, audience_percent, critic_percent):
	x_data = pd.DataFrame({'audience_average':audience_average, 'critic_average':critic_average,
	'audience_percent':audience_percent, 'critic_percent':critic_percent})
	x_melt = pd.melt(x_data)
	posthoc = pairwise_tukeyhsd(
    x_melt['value'], x_melt['variable'], alpha=0.05)
	print(posthoc)
	fig = posthoc.plot_simultaneous()
	plt.xlabel('rating')
	plt.show()




def main():
	wiki_movie_df, rating_df, genres_df, wiki_genres_df = process_data.read_data()
	seaborn.set()
	#audience_rating, critic_rating = get_rating(rating_df)
	audience_average, critic_average, audience_percent, critic_percent = get_data(rating_df)
	pvalue = do_anova(audience_average, critic_average, audience_percent, critic_percent)
	if (pvalue < 0.05):
		print(" \n ")
		print("Do post hoc Tukey test")
	do_post_hoc(audience_average, critic_average, audience_percent, critic_percent)



if __name__ == "__main__":
	main()
