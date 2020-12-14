import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#helper functions. use them when needed
def get_title_from_index(index):
    return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
    return df[df.title == title]["index"].values[0]

#step 1: read csv file
df = pd.read_csv("movie_dataset.csv")
#print(df.columns)

#step 2: select features
features = ['keywords','cast','genres','director']

#step 3: create a column in df which combines all selected features
#replaced all na(which is float) with an empty string so that it could be combined withother strings in the function
for feature in features:
    df[feature] = df[feature].fillna('')

def combine_features(row):
    return row['keywords']+" "+row['cast']+" "+row['genres']+" "+row['director']

#call the function therefore used apply and combine vertically therefore used axis=1
df["combined_features"] = df.apply(combine_features,axis=1)

#print("Combined Features:", df["combined_features"].head())

#step 4: create count matrix from this new combined column
cv = CountVectorizer()

count_matrix = cv.fit_transform(df["combined_features"])

#step 5: compute the cosine similarity based on the count_matrix
cosine_sim = cosine_similarity(count_matrix)
movie_user_likes = "Avatar"

#step 6: get index of this movie from its title
movie_index = get_index_from_title(movie_user_likes)

similar_movies = list(enumerate(cosine_sim[movie_index]))

#step 7: get a list of similar movies in descending order of similarity score
sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)

#step 8: print titles of first 50 movies
i=0
for movie in sorted_similar_movies:
    print (get_title_from_index(movie[0]))
    i=i+1
    if i>50:
        break