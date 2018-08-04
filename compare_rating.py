import process_data
import pandas as pd
import numpy as np
import sys
from scipy import stats
import matplotlib.pyplot as plt
import seaborn


def get_rating(rating_df):
	rating_df = rating_df.dropna(subset = ['audience_average', 'critic_average'])
	#print(rating_df.count())
	rating_df = rating_df.sort_values(by=['audience_ratings'], ascending = False)
	rating_df = rating_df[:300]
	#print(rating_df.count())
	# Because the audience rating is out of 5 and critic rating is out of 10, the stardant of them are different
	rating_df['critic_average'] = rating_df['critic_average'] /2
	#print(critic_rating)
	rating_df['audience_average'] = rating_df['audience_average'].round(1)
	rating_df['critic_average'] = rating_df['critic_average'].round(1)
	#print(audience_rating)
	#print(critic_rating)
	audience_rating = rating_df['audience_average']
	critic_rating = rating_df['critic_average']
	# df = rating_df[sample(nrow(rating_df), 200), ]
	# audience_rating = df['audience_average']
	# critic_rating = df['critic_average']
	# print(df)
	print("----- Data -----")
	print(rating_df.count())
	print(" \n ")
	print(audience_rating.head())
	print(" \n ")
	print(critic_rating.head())
	return audience_rating, critic_rating




def check_t_test(audience_rating, critic_rating):
	# Test thry are normalization
	test_audience_norm = stats.normaltest(audience_rating).pvalue
	test_critic_norm = stats.normaltest(critic_rating).pvalue
	#print(test_audience_norm, test_critic_norm)
	#print(test_audience_norm.pvalue)
	#print(test_critic_norm.pvalue)

	# Test their if have equal variance
	equal_variance = stats.levene(audience_rating,critic_rating).pvalue
	#print (equal_variance.pvalue)
	return test_audience_norm, test_critic_norm, equal_variance



def do_log(num):
	return (np.log(num))

def do_exp(num):
	return (np.exp(num))

def do_sqrt(num):
	return (np.sqrt(num))

def do_times(num):
	return num**2



def t_test(audience_rating, critic_rating):
	plt.figure(1)                # the first figure
	plt.subplot(211)
	plt.xlabel('Audience rating')
	plt.ylabel('number')
	plt.hist(audience_rating, bins='auto')
	plt.subplot(212)
	plt.xlabel('Critic rating')
	plt.ylabel('number')
	plt.hist(critic_rating, bins='auto')
	plt.show()

	test_audience_norm, test_critic_norm, equal_variance = check_t_test(audience_rating, critic_rating)
	print("pvalue of audience rating normality: ",test_audience_norm)
	print("pvalue of critic rating normality: ",test_critic_norm)
	print("pvalue of equal variance: ", equal_variance)
	# _, ttest_pvalue = stats.ttest_ind(audience_rating, critic_rating)
	# print("!!!!!!", ttest_pvalue)

	if (audience_rating.count() >= 40 and critic_rating.count() >= 40):
		print("Because the data n >= 40, it may ok for T-test")
		ttest = stats.ttest_ind(audience_rating, critic_rating)
		print("mean of audience rating: ", audience_rating.mean())
		print("mean of critic rating: ", critic_rating.mean())
		print("pvalue of T-test: ", ttest.pvalue)
		if (ttest.pvalue < 0.05):
			print("Because pvalue < 0.05, they have different means")
	else:
		print("Cannot use T-test")




def u_test(audience_rating, critic_rating):
	# audience_rating = audience_rating.sort_values(ascending=True)
	# critic_rating = critic_rating.sort_values(ascending=True)
	#print(critic_rating)
	u_pvalue = stats.mannwhitneyu(audience_rating, critic_rating).pvalue
	print("pvalue of U-test: ", u_pvalue)
	if (u_pvalue < 0.05):
		print("Because pvalue < 0.05, they have different means")



def regression(audience_rating, critic_rating):
	# reg = stats.linregress(audience_rating, critic_rating)
	# plt.xlabel('Audience rating')
	# plt.ylabel('critic_rating')
	# plt.scatter(audience_rating, critic_rating)
	# plt.show()
	# print("slope pf linear regression: ", reg.slope)
	# print("intercept of linear regression: ", reg.intercept)
	# print("pvalue of linear regression: ", reg.pvalue)

	reg = stats.linregress(audience_rating, critic_rating)
	predict = audience_rating*reg.slope + reg.intercept
	print(reg)
	#print("the pvalue of fit is: ", fit.pvalue)
	plt.figure(2)
	plt.title('audience_rating vs critic_rating')
	plt.xlabel('audience_rating')
	plt.ylabel('critic_rating')
	plt.plot(audience_rating,critic_rating,'b.', alpha=0.5)
	plt.plot(audience_rating,predict,'r-', linewidth=3)
	plt.show()

	residuals = critic_rating - (reg.slope*audience_rating + reg.intercept)

	plt.title('Histogram of the residuals ')
	plt.xlabel('Residuals')
	plt.ylabel('Frequency')
	plt.hist(residuals, density=True, bins=30)
	plt.show()
	print("correlation coefficient r: ", reg.rvalue)
	print("r squared: ", reg.rvalue**2 )



def main():
	wiki_movie_df, rating_df, genres_df, wiki_genres_df = process_data.read_data()
	audience_rating, critic_rating = get_rating(rating_df)
	seaborn.set()
	#plt.savefig('rating.png')

	# do T-test for testing if audience rating and critic norm have the same means
	print("\n")
	print("----- T-test -----")
	t_test(audience_rating, critic_rating)
	print("\n")
	print("----- U-test -----")
	u_test(audience_rating, critic_rating)
	print("\n")
	print("----- Regression -----")
	regression(audience_rating, critic_rating)





if __name__ == "__main__":
	main()
