import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def cosine_similarity(x,y):
    if len(x) != len(y):
        return none
    dot_product = np.dot(x,y)
    x_magnitude = np.sqrt(sum(x**2))
    y_magnitude = np.sqrt(sum(y**2))

    if x_magnitude == 0 and y_magnitude == 0:
        return 0
    return dot_product / (x_magnitude * y_magnitude)

corpus = [
        'I am going to the store because I lost my keys',
        'The manager and his team worked very hard to complete the project on time',
        'Ankur was happy because he got a gift'
         ]    
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(corpus).toarray()
print("Word Frequency Vectors:\n",x)

d1_d2 = cosine_similarity(x[0],x[1])
d1_d3 = cosine_similarity(x[0],x[2])
d2_d3 = cosine_similarity(x[1],x[2])

print("Cosine Similarity\n")
print("Relation between doc 1 and 2",d1_d2)
print("Relation between doc 1 and 3",d1_d3)
print("Relation between doc 2 and 3",d2_d3)








