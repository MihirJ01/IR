from nltk.corpus import stopwords
from nltk import word_tokenize

sen = "this is a cat it is in black "

stopword = set(stopwords.words('english'))

tokens = word_tokenize(sen)

filteredsen = []
for token in tokens:
    if token not in stopword:
        filteredsen.append(token)
print(filteredsen)


