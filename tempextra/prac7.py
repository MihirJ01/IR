from transformers import pipeline

# Load the summarization pipeline
summarizer = pipeline("summarization", model="Falconsai/text_summarization")

# Define the document to be summarized
ARTICLE = """
Hugging Face: Revolutionizing Natural Language Processing

Introduction

In the rapidly evolving field of Natural Language Processing (NLP), Hugging Face has emerged as a prominent and innovative force. This article will explore the story and significance of Hugging Face, a company that has made remarkable contributions to NLP and AI as a whole. From its inception to its role in democratizing AI, Hugging Face has left an indelible mark on the industry.
"""

# Get the summary
summary = summarizer(ARTICLE, max_length=1000, min_length=30, do_sample=False)

# Print the summary
print("Summary:")
print(summary[0]['summary_text'])
