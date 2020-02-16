# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # Before your start:
# - Read the README.md file
# - Comment as much as you can and use the resources in the README.md file
# - Happy learning!

# %%
# Import your libraries:
#%%
get_ipython().magic('matplotlib inline')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# %% [markdown]
# In this lab, we will explore a dataset that describes websites with different features and labels them either benign or malicious . We will use supervised learning algorithms to figure out what feature patterns malicious websites are likely to have and use our model to predict malicious websites.
# 
# # Challenge 1 - Explore The Dataset
# 
# Let's start by exploring the dataset. First load the data file:

# %%
#%%
websites = pd.read_csv('../website.csv')

# %% [markdown]
# #### Explore the data from an bird's-eye view.
# 
# You should already been very familiar with the procedures now so we won't provide the instructions step by step. Reflect on what you did in the previous labs and explore the dataset.
# 
# Things you'll be looking for:
# 
# * What the dataset looks like?
# * What are the data types?
# * Which columns contain the features of the websites?
# * Which column contains the feature we will predict? What is the code standing for benign vs malicious websites?
# * Do we need to transform any of the columns from categorical to ordinal values? If so what are these columns?
# 
# Feel free to add additional cells for your explorations. Make sure to comment what you find out.

# %%
# Your code here
plt.matshow(websites.corr())
plt.show()

# %% [markdown]
# Data Description:
# 
# URL: it is the anonimous identification of the URL analyzed in the study
# 
# URL_LENGTH: it is the number of characters in the URL
# 
# NUMBERSPECIALCHARACTERS: it is number of special characters identified in the URL, such as, “/”, “%”, “#”, “&”, “. “, “=”
# 
# CHARSET: it is a categorical value and its meaning is the character encoding standard (also called character set).
# 
# SERVER: it is a categorical value and its meaning is the operative system of the server got from the packet response.
# 
# CONTENT_LENGTH: it represents the content size of the HTTP header.
# 
# WHOIS_COUNTRY: it is a categorical variable, its values are the countries we got from the server response (specifically, our script used the API of Whois).
# 
# WHOIS_STATEPRO: it is a categorical variable, its values are the states we got from the server response (specifically, our script used the API of Whois).
# 
# WHOIS_REGDATE: Whois provides the server registration date, so, this variable has date values with format DD/MM/YYY HH:MM
# 
# WHOISUPDATEDDATE: Through the Whois we got the last update date from the server analyzed
# 
# TCPCONVERSATIONEXCHANGE: This variable is the number of TCP packets exchanged between the server and our honeypot client
# 
# DISTREMOTETCP_PORT: it is the number of the ports detected and different to TCP
# 
# REMOTE_IPS: this variable has the total number of IPs connected to the honeypot
# 
# APP_BYTES: this is the number of bytes transfered
# 
# SOURCEAPPPACKETS: packets sent from the honeypot to the server
# 
# REMOTEAPPPACKETS: packets received from the server
# 
# APP_PACKETS: this is the total number of IP packets generated during the communication between the honeypot and the server
# 
# DNSQUERYTIMES: this is the number of DNS packets generated during the communication between the honeypot and the server
# 
# TYPE: this is a categorical variable, its values represent the type of web page analyzed, specifically, 1 is for malicious websites and 0 is for benign websites
# 

# %%
websites.dtypes


# %%
#%%
websites.isnull().sum()


# %%
#%%
websites


# %%
# Your comment here
# Se realizara limpieza de dataframe tratando de conservar la mayor información posible.

# %% [markdown]
# #### Next, evaluate if the columns in this dataset are strongly correlated.
# 
# In the Mushroom supervised learning lab we did recently, we mentioned we are concerned if our dataset has strongly correlated columns because if it is the case we need to choose certain ML algorithms instead of others. We need to evaluate this for our dataset now.
# 
# Luckily, most of the columns in this dataset are ordinal which makes things a lot easier for us. In the next cells below, evaluate the level of collinearity of the data.
# 
# We provide some general directions for you to consult in order to complete this step:
# 
# 1. You will create a correlation matrix using the numeric columns in the dataset.
# 
# 1. Create a heatmap using `seaborn` to visualize which columns have high collinearity.
# 
# 1. Comment on which columns you might need to remove due to high collinearity.

# %%
# Your code here
#%%
websites.corr()


# %%
#%%
sns.heatmap(websites.corr())


# %%
# Your comment here
#Hay que botar las columnas de URL_LENGTH, NUMBER_SPECIAL_CHARACTERS, CONTENT_LENGTH, de momento parece que no tienen mucha relación
# con lo que se necesita y se continuara de ese modo.

# %% [markdown]
# # Challenge 2 - Remove Column Collinearity.
# 
# From the heatmap you created, you should have seen at least 3 columns that can be removed due to high collinearity. Remove these columns from the dataset.
# 
# Note that you should remove as few columns as you can. You don't have to remove all the columns at once. But instead, try removing one column, then produce the heatmap again to determine if additional columns should be removed. As long as the dataset no longer contains columns that are correlated for over 90%, you can stop. Also, keep in mind when two columns have high collinearity, you only need to remove one of them but not both.
# 
# In the cells below, remove as few columns as you can to eliminate the high collinearity in the dataset. Make sure to comment on your way so that the instructional team can learn about your thinking process which allows them to give feedback. At the end, print the heatmap again.

# %%
# Los reviales las debiles: URL_LENGTH, NUMBER_SPECIAL_CHARACTERS, CONTENT_LENGTH.
#%%
sns.heatmap(websites[['URL', 'CHARSET', 'SERVER',
       'WHOIS_COUNTRY', 'WHOIS_STATEPRO', 'WHOIS_REGDATE',
       'WHOIS_UPDATED_DATE', 'TCP_CONVERSATION_EXCHANGE',
       'DIST_REMOTE_TCP_PORT', 'REMOTE_IPS', 'APP_BYTES', 'SOURCE_APP_PACKETS',
       'REMOTE_APP_PACKETS', 'SOURCE_APP_BYTES', 'REMOTE_APP_BYTES',
       'APP_PACKETS', 'DNS_QUERY_TIMES', 'Type']].corr())
websites


# %%
# Your comment here
#%%
try:
    websites.drop(columns=['URL_LENGTH','NUMBER_SPECIAL_CHARACTERS','CONTENT_LENGTH'], axis=1, inplace=True)
except:
    pass
print('holi')


# %%
# Print heatmap again
#%%
sns.heatmap(websites.corr(), annot=True, linewidths=.5, cbar=False);

# %% [markdown]
# # Challenge 3 - Handle Missing Values
# 
# The next step would be handling missing values. **We start by examining the number of missing values in each column, which you will do in the next cell.**

# %%
# Your code here
#%%
websites.isnull().sum()

# %% [markdown]
# If you remember in the previous labs, we drop a column if the column contains a high proportion of missing values. After dropping those problematic columns, we drop the rows with missing values.
# 
# #### In the cells below, handle the missing values from the dataset. Remember to comment the rationale of your decisions.

# %%
# Your code here
#%%
websites.dropna(axis=0, inplace=True)


# %%
# Your comment here

# %% [markdown]
# #### Again, examine the number of missing values in each column. 
# 
# If all cleaned, proceed. Otherwise, go back and do more cleaning.

# %%
# Examine missing values in each column
websites.isnull().sum()

# %% [markdown]
# # Challenge 4 - Handle `WHOIS_*` Categorical Data
# %% [markdown]
# There are several categorical columns we need to handle. These columns are:
# 
# * `URL`
# * `CHARSET`
# * `SERVER`
# * `WHOIS_COUNTRY`
# * `WHOIS_STATEPRO`
# * `WHOIS_REGDATE`
# * `WHOIS_UPDATED_DATE`
# 
# How to handle string columns is always case by case. Let's start by working on `WHOIS_COUNTRY`. Your steps are:
# 
# 1. List out the unique values of `WHOIS_COUNTRY`.
# 1. Consolidate the country values with consistent country codes. For example, the following values refer to the same country and should use consistent country code:
#     * `CY` and `Cyprus`
#     * `US` and `us`
#     * `SE` and `se`
#     * `GB`, `United Kingdom`, and `[u'GB'; u'UK']`
# 
# #### In the cells below, fix the country values as intructed above.

# %%
# Your code here
#websites.WHOIS_COUNTRY.unique()     #[13] "[u'GB'; u'UK']"
websites.replace(to_replace=['UK','United Kingdom',"[u'GB'; u'UK']"], value='GB', inplace=True)


# %%
#%%
websites.replace(to_replace=['Cyprus'], value='CY', inplace=True)


# %%
websites.WHOIS_COUNTRY.unique()

# %% [markdown]
# Since we have fixed the country values, can we convert this column to ordinal now?
# 
# Not yet. If you reflect on the previous labs how we handle categorical columns, you probably remember we ended up dropping a lot of those columns because there are too many unique values. Too many unique values in a column is not desirable in machine learning because it makes prediction inaccurate. But there are workarounds under certain conditions. One of the fixable conditions is:
# 
# #### If a limited number of values account for the majority of data, we can retain these top values and re-label all other rare values.
# 
# The `WHOIS_COUNTRY` column happens to be this case. You can verify it by print a bar chart of the `value_counts` in the next cell to verify:

# %%
# Your code here
websites.WHOIS_COUNTRY.value_counts().plot.bar();

# %% [markdown]
# #### After verifying, now let's keep the top 10 values of the column and re-label other columns with `OTHER`.

# %%
# Your code here
websites.WHOIS_COUNTRY.str.upper().value_counts()
top10 = ['US','NONE','CA','ES','GB','AU','PA','JP','CN','IN']

for y in range(len(websites.WHOIS_COUNTRY)):
    if websites.WHOIS_COUNTRY[y] not in top10:
        websites[y] = 'OTHER'
    else:
        continue

websites.WHOIS_COUNTRY.unique()

# %% [markdown]
# Now since `WHOIS_COUNTRY` has been re-labelled, we don't need `WHOIS_STATEPRO` any more because the values of the states or provinces may not be relevant any more. We'll drop this column.
# 
# In addition, we will also drop `WHOIS_REGDATE` and `WHOIS_UPDATED_DATE`. These are the registration and update dates of the website domains. Not of our concerns.
# 
# #### In the next cell, drop `['WHOIS_STATEPRO', 'WHOIS_REGDATE', 'WHOIS_UPDATED_DATE']`.

# %%
# Your code here

# %% [markdown]
# # Challenge 5 - Handle Remaining Categorical Data & Convert to Ordinal
# 
# Now print the `dtypes` of the data again. Besides `WHOIS_COUNTRY` which we already fixed, there should be 3 categorical columns left: `URL`, `CHARSET`, and `SERVER`.

# %%
# Your code here

# %% [markdown]
# #### `URL` is easy. We'll simply drop it because it has too many unique values that there's no way for us to consolidate.

# %%
# Your code here

# %% [markdown]
# #### Print the unique value counts of `CHARSET`. You see there are only a few unique values. So we can keep it as it is.

# %%
# Your code here

# %% [markdown]
# `SERVER` is a little more complicated. Print its unique values and think about how you can consolidate those values.
# 
# #### Before you think of your own solution, don't read the instructions that come next.

# %%
# Your code here

# %% [markdown]
# ![Think Hard](../think-hard.jpg)

# %%
# Your comment here

# %% [markdown]
# Although there are so many unique values in the `SERVER` column, there are actually only 3 main server types: `Microsoft`, `Apache`, and `nginx`. Just check if each `SERVER` value contains any of those server types and re-label them. For `SERVER` values that don't contain any of those substrings, label with `Other`.
# 
# At the end, your `SERVER` column should only contain 4 unique values: `Microsoft`, `Apache`, `nginx`, and `Other`.

# %%
# Your code here


# %%
# Count `SERVER` value counts here

# %% [markdown]
# OK, all our categorical data are fixed now. **Let's convert them to ordinal data using Pandas' `get_dummies` function ([documentation](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html)).** Make sure you drop the categorical columns by passing `drop_first=True` to `get_dummies` as we don't need them any more. **Also, assign the data with dummy values to a new variable `website_dummy`.**

# %%
# Your code here

# %% [markdown]
# Now, inspect `website_dummy` to make sure the data and types are intended - there shouldn't be any categorical columns at this point.

# %%
# Your code here

# %% [markdown]
# # Challenge 6 - Modeling, Prediction, and Evaluation
# 
# We'll start off this section by splitting the data to train and test. **Name your 4 variables `X_train`, `X_test`, `y_train`, and `y_test`. Select 80% of the data for training and 20% for testing.**

# %%
from sklearn.model_selection import train_test_split

# Your code here:

# %% [markdown]
# #### In this lab, we will try two different models and compare our results.
# 
# The first model we will use in this lab is logistic regression. We have previously learned about logistic regression as a classification algorithm. In the cell below, load `LogisticRegression` from scikit-learn and initialize the model.

# %%
# Your code here:


# %% [markdown]
# Next, fit the model to our training data. We have already separated our data into 4 parts. Use those in your model.

# %%
# Your code here:


# %% [markdown]
# finally, import `confusion_matrix` and `accuracy_score` from `sklearn.metrics` and fit our testing data. Assign the fitted data to `y_pred` and print the confusion matrix as well as the accuracy score

# %%
# Your code here:


# %% [markdown]
# What are your thoughts on the performance of the model? Write your conclusions below.

# %%
# Your conclusions here:


# %% [markdown]
# #### Our second algorithm is is K-Nearest Neighbors. 
# 
# Though is it not required, we will fit a model using the training data and then test the performance of the model using the testing data. Start by loading `KNeighborsClassifier` from scikit-learn and then initializing and fitting the model. We'll start off with a model where k=3.

# %%
# Your code here:


# %% [markdown]
# To test your model, compute the predicted values for the testing sample and print the confusion matrix as well as the accuracy score.

# %%
# Your code here:


# %% [markdown]
# #### We'll create another K-Nearest Neighbors model with k=5. 
# 
# Initialize and fit the model below and print the confusion matrix and the accuracy score.

# %%
# Your code here:


# %% [markdown]
# Did you see an improvement in the confusion matrix when increasing k to 5? Did you see an improvement in the accuracy score? Write your conclusions below.

# %%
# Your conclusions here:


# %% [markdown]
# # Bonus Challenge - Feature Scaling
# 
# Problem-solving in machine learning is iterative. You can improve your model prediction with various techniques (there is a sweetspot for the time you spend and the improvement you receive though). Now you've completed only one iteration of ML analysis. There are more iterations you can conduct to make improvements. In order to be able to do that, you will need deeper knowledge in statistics and master more data analysis techniques. In this bootcamp, we don't have time to achieve that advanced goal. But you will make constant efforts after the bootcamp to eventually get there.
# 
# However, now we do want you to learn one of the advanced techniques which is called *feature scaling*. The idea of feature scaling is to standardize/normalize the range of independent variables or features of the data. This can make the outliers more apparent so that you can remove them. This step needs to happen during Challenge 6 after you split the training and test data because you don't want to split the data again which makes it impossible to compare your results with and without feature scaling. For general concepts about feature scaling, click [here](https://en.wikipedia.org/wiki/Feature_scaling). To read deeper, click [here](https://medium.com/greyatom/why-how-and-when-to-scale-your-features-4b30ab09db5e).
# 
# In the next cell, attempt to improve your model prediction accuracy by means of feature scaling. A library you can utilize is `sklearn.preprocessing.RobustScaler` ([documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html)). You'll use the `RobustScaler` to fit and transform your `X_train`, then transform `X_test`. You will use logistic regression to fit and predict your transformed data and obtain the accuracy score in the same way. Compare the accuracy score with your normalized data with the previous accuracy data. Is there an improvement?

# %%
# Your code here

