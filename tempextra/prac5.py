from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Sample corpus
documents = [
    "The cat sat on the mat.",
    "The dog barked at the cat.",
    "I love programming in Python.",
    "Python is a great programming language.",
    "Dogs and cats are popular pets.",
    "I enjoy solving coding challenges in Python."
]

# Step 1: Convert text to TF-IDF features
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(documents)

# Step 2: Apply K-Means clustering
num_clusters = 2  # Define the number of clusters
kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
kmeans.fit(tfidf_matrix)

# Step 3: Assign documents to clusters
labels = kmeans.labels_

# Display results
for i, doc in enumerate(documents):
    print(f"Document {i + 1}: {doc}")
    print(f"Cluster: {labels[i]}\n")

# Step 4: Evaluate clustering performance
silhouette_avg = silhouette_score(tfidf_matrix, labels)
print(f"Silhouette Score: {silhouette_avg:.3f}")
