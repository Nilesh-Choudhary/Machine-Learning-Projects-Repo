# %%
# Importing the necessary libraries
import pandas as pd
import numpy as np
import plotly.express as px

from textblob import TextBlob
from wordcloud import WordCloud,STOPWORDS


# %%
# Reading the file 
sentiment_df=pd.read_csv('E:\PythonProjects\DataSet\mobile_reviews.csv')
sentiment_df

# %%
# Example
TextBlob('India is a  country').sentiment

# %%
# Function to identify polarity
def polarity_identify(text):
   return TextBlob(text).sentiment.polarity

# %%
# Applying polarity_identify function to each element of Brief Review Column to derive new column polarity

sentiment_df['polarity']=sentiment_df['Brief Review'].apply(polarity_identify)

# %%
# Function to show Positive , Negative or Neutral Review On the basis of polarity score

def get_review_type(polarity):
  if polarity < 0:
    return "Negative"
  elif polarity > 0:
    return "Positive"
  else:
    return "Neutral"

# %%
# Applying get_review_type function to each element of polarity column to derive new column ReviewType
sentiment_df['Review_Type'] = sentiment_df['polarity'].apply(get_review_type)

# %%
# Grouping to identify count of Positive, Negative and Neutral Reviews
count_reviews_df=sentiment_df.groupby('Review_Type',as_index=False)['User ID'].count()

# %%
# Renaming Column
count_reviews_df.rename(columns={'User ID':'Count of Users'},inplace=True)

# %%
# Plotting bar chart to depict count of reviews (Positive,Negative and Neutral)
fig=px.bar(count_reviews_df,x='Review_Type',y='Count of Users',color='Review_Type',template='plotly_dark',color_discrete_sequence=['Red','Blue','Green'])
fig.update_layout(width=800,title=dict(text='Sentiment Analysis of User Reviews',x=0.5))
fig.show()

# %%
# Creating Seperate Dataframes for positive and negative reviews
neg_review_df=sentiment_df[sentiment_df['Review_Type']=='Negative']
pos_review_df=sentiment_df[sentiment_df['Review_Type']=='Positive']

# %%
# Concatenating the elements  
neg_reviews_text = ' '.join(neg_review_df['Brief Review'])
pos_reviews_text = ' '.join(pos_review_df['Brief Review'])

# %%
# Concatenating all the reviews
all_reviews_text = ' '.join(sentiment_df['Brief Review'])

# %%
# Plotting wordcloud based on postive reviews
stopwords=STOPWORDS
pos_wordcloud = WordCloud(stopwords=stopwords,width=800).generate(pos_reviews_text)
fig=px.imshow(pos_wordcloud,template='plotly_dark')
fig.update_layout(xaxis_visible=False,yaxis_visible=False)
fig.show()

# %%
# Plotting wordcloud based on negative reviews
stopwords=STOPWORDS
neg_wordcloud = WordCloud(stopwords=stopwords,width=800).generate(neg_reviews_text)
fig=px.imshow(neg_wordcloud,template='plotly_dark')
fig.update_layout(xaxis_visible=False,yaxis_visible=False)
fig.show()

# %%
# Plotting wordcloud based on the basis of total reviews
stopwords=STOPWORDS
all_wordcloud = WordCloud(stopwords=stopwords,width=800).generate(all_reviews_text)
fig=px.imshow(all_wordcloud,template='plotly_dark')
fig.update_layout(xaxis_visible=False,yaxis_visible=False)
fig.show()


