from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

documents = [
    "I love the taste of Italian food, especially pizza and pasta.",
    "Artificial Intelligence and Machine Learning are revolutionizing technology.",
    "Football is a popular sport around the world, especially in Europe.",
    "I enjoy coding and building software solutions with Python.",
    "The economy is affected by inflation and the stock market.",
    "The Python programming language is widely used for data science and AI."
]

vectorizer = TfidfVectorizer(stop_words="english")

x = vectorizer.fit_transform(documents)

num_cluster = 3

kmeans = KMeans(n_clusters = num_cluster,random_state=42)
kmeans.fit(x)

labels = kmeans.labels_

print(f"Document 1: {documents[0]}")
print(f"Cluster: {labels[0]}\n")

print(f"Document 2: {documents[1]}")
print(f"Cluster: {labels[1]}\n")

print(f"Document 3: {documents[2]}")
print(f"Cluster: {labels[2]}\n")

print(f"Document 4: {documents[3]}")
print(f"Cluster: {labels[3]}\n")

print(f"Document 5: {documents[4]}")
print(f"Cluster: {labels[4]}\n")

print(f"Document 6: {documents[5]}")
print(f"Cluster: {labels[5]}\n")

silhouette_avg = silhouette_score(x,labels)
print(silhouette_avg)
