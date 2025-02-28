import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

corpus=[
    'this is the first document.',
    'this document is second document.',
    'and this is third one.',
    'is this first document?'
    ]

vec = CountVectorizer()
x = vec.fit_transform(corpus)
print(x.toarray())

df = pd.DataFrame(x.toarray(),columns=vec.get_feature_names_out())
print(df)

andData = df[(df['this']==1) & (df['first']==1)]
print("Indices in which this and first are present at",andData.index.tolist())
