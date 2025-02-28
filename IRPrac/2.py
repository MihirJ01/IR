import nltk
from nltk.util import ngrams
from nltk import word_tokenize

text = "This is a sample text for unigram, bigram, and trigram extraction using NLTK,"

lt = text.lower()

tokens = word_tokenize(lt)
print("OG text")
print(tokens)

unigrams = list(ngrams(tokens,1))
print("unigram")
print(unigrams)

bigram = list(ngrams(tokens,2))
print("Bigram")
print(bigram)

trigram = list(ngrams(tokens,3))
print("Trigram")
print(trigram)
