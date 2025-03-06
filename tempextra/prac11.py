import requests
from bs4 import BeautifulSoup

url = "https://google.com"

# Send HTTP GET request
response = requests.get(url)

# Check if request was successful
if response.status_code == 200:  # Use '==' instead of '='
    soup = BeautifulSoup(response.text, "html.parser")  # Parse HTML content
    text_content = soup.get_text()  # Extract all text from the page
    print(text_content)  # Print the extracted text
else:
    print(f"Error: Unable to fetch content. Status code: {response.status_code}")
