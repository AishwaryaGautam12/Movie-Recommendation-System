#To count the number of times london and paris have appeared in the text and then find cosine similarity using scikit learn

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

text = ["London Paris London", "Paris Paris London"]

#CountVectorizer() is a class
cv = CountVectorizer() 

count_matrix = cv.fit_transform(text)

#print (count_matrix.toarray())

similarity_scores = cosine_similarity(count_matrix)

print (similarity_scores)