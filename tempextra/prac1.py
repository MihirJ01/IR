import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Define the corpus
corpus = [
    'this is the first document.',
    'this document is the second document.',
    'and this is the third one.',
    'is this the first document?'
]

# Initialize the CountVectorizer
vectorizer = CountVectorizer()

# Fit and transform the corpus
X = vectorizer.fit_transform(corpus)

# Print the fit-transformed array
print("Fit transform is:")
print(X.toarray())

# Create a DataFrame
df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

print("The generated DataFrame is:")
print(df)

# Get indices where 'this' and 'first' terms are present
alldata = df[(df['this'] == 1) & (df['first'] == 1)]
print("Indices where 'this' and 'first' terms are present are:", alldata.index.tolist())

# Get indices where either 'this' or 'first' terms are present
ordat = df[(df['this'] == 1) | (df['first'] == 1)]
print("Indices where either of 'this' or 'first' terms are present are:", ordat.index.tolist())

# Get indices where 'and' term is not present
notdata = df[(df['and'] != 1)]
print("Indices where 'and' term is not present are:", notdata.index.tolist())
