import nltk
from nltk import word_tokenize
from nltk.util import ngrams

# Sample text
text = "This is a sample text for unigram, bigram, and trigram extraction using NLTK."

# Tokenize and convert to lowercase
tokens = word_tokenize(text.lower())

# Generate unigrams, bigrams, and trigrams
unigrams = list(ngrams(tokens, 1))
bigrams = list(ngrams(tokens, 2))
trigrams = list(ngrams(tokens, 3))

# Print results
print("Original Text:", text)
print("\nUnigrams:", unigrams)
print("\nBigrams:", bigrams)
print("\nTrigrams:", trigrams)
