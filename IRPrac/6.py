import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# Example sentence
sentence = "This is a simple example to demonstrate stop word removal."

# Tokenize the sentence
words = word_tokenize(sentence)

# Get English stop words
stop_words = set(stopwords.words('english'))

# Remove stop words
filtered_sentence = []
     for word in words:
         if word.lower() not in stop_words
             filtered_sentence.append(word)

# Print the result
print("Original Sentence:", sentence)
print("After Stopword Removal:", " ".join(filtered_sentence))
