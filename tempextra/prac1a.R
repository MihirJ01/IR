library(tm)
library(dplyr)

# Create a text corpus
corpus <- Corpus(VectorSource(c(
  'this is the first document.',
  'this document is second document.',
  'and this is third one.' ,
  'is this first document?'
)))

# Create Document-Term Matrix
dtm <- DocumentTermMatrix(corpus)
matrix_data <- as.matrix(dtm)

# Convert to DataFrame
df <- as.data.frame(matrix_data)

print("Term Frequency Matrix:")
print(df)

# Ensure column names exist before filtering
if ("this" %in% colnames(df) & "first" %in% colnames(df)) {
  alldata <- df %>% filter(this == 1 & first == 1)
  print("Rows where 'this' and 'first' terms are present: ")
  print(rownames(alldata))
} else {
  print("'this' or 'first' column is missing in the term matrix.")
}

if ("this" %in% colnames(df) & "first" %in% colnames(df)) {
  ordata <- df %>% filter(this == 1 | first == 1)
  print("Rows where either 'this' or 'first' terms are present: ")
  print(rownames(ordata))
} else {
  print("'this' or 'first' column is missing in the term matrix.")
}

if ("and" %in% colnames(df)) {
  notdata <- df %>% filter(and != 1)
  print("Rows where 'and' term is not present:")
  print(rownames(notdata))
} else {
  print("'and' column is missing in the term matrix.")
}
