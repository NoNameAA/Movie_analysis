# CMPT353

This project is for analyzing movies and there are five complete files.

Document:
1. process_data.py
    Running Command: python3 process_data.py
    input: wikidata-movies.json.gz
           rotten-tomatoes.json.gz
           genres.json.gz
           omdb-data.json.gz
    library: import pandas as pd
             import sys
    Expected: Get data from different input files and do help for processing and analyzing data later
    Output: nothing


2. compare_rating.py
    Running Command: python3 compare_rating.py
    input: process_data.py
           rotten-tomatoes.json.gz
    library: import pandas as pd
             import sys
             import numpy as np   
             from scipy import stats
             import matplotlib.pyplot as plt 
             import seaborn  
    Expected: Do statistic test (T-test, U-Test and ordinary least squares) for audience_average and critic_average. Discuss if audience reviews and critic reviews have the same means or not.
    OutPut: results of statistic test, two histograms and a scatter plots


3. compare_audi_crit.py
    Running Command: python3 compare_audi_crit.py
    input: process_data.py
           rotten-tomatoes.json.gz
    library: import pandas as pd
             import sys
             import numpy as np   
             from scipy import stats
             import matplotlib.pyplot as plt  
             import seaborn
             from statsmodels.stats.multicomp import pairwise_tukeyhsd
    Expected: Do statistic test (anova and post hoc Tukey test) for multiple values. Discuss if these values audience reviews, critic reviews, audience_percent and critic_percent can be concluded to have different means.
    Output: results of statistic test, a table, one histogram and a spiffy plot


4. movie_predict.py (There is a jupyter notebook for demo this code due to too many figures and tables)
    Running Command: python3 movie_predict.py
    Input: process_data.py
           wikidata-movies.json.gz
           rotten-tomatoes.json.gz
           genres.json.gz
           omdb-data.json.gz
    Library: warnings, process_data.py, pandas, numpy, sys, sklearn, matplotlib, seaborn
    Expected: Using rating data and made_profit to train machine learning.
    Output: accuracy of models, tables, correlation heatmap


5. NLP_predict.py (This is not a submitted code, it is a nice try!)
    Running Command: python3 NLP_predict.py
    Input data: omdb-data.json.gz
    Library: warnings, numpy, pandas, sklearn, nltk, string, re
    Expected: Using description of movie to predict the genres of movies.
    Output: accuracy of prediction



Order of execution:
  TA and prof do not need to execute the file "process_data.py", because this is a help file which has no output. The purpose of the file is getting data from four data sources and doing help for other files. The other four files "compare_rating.py", "compare_audi_crit.py", "movie_predict.py" and "NLP_predict.py" are individual files, so there is no necessary execution order between them.

