import sys
import os
import time
import logging
from flask import Flask, render_template, request, jsonify, url_for, send_file
import pandas as pd
from gtts import gTTS
import re
from transformers import pipeline
from rapidfuzz import fuzz
import random
import spacy
from g2p_en import G2p
import requests
from bs4 import BeautifulSoup  # For web scraping
from flask import Flask, render_template, request
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
import time

from flask import Flask, render_template, request


sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))





USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
]



headers = {"User-Agent": random.choice(USER_AGENTS)}

conversation_history = []
current_topic = None

# Path to your CSV files
vocab_csv_path = 'static/vocabulary.csv'
response_csv_path = 'static/responses.csv'

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize NLP pipeline and phoneme extractor
nlp = pipeline("ner", model="dslim/bert-base-NER", tokenizer="dslim/bert-base-NER")
g2p = G2p()




# Clean and initialize vocabulary CSV
if not os.path.exists(vocab_csv_path):
    pd.DataFrame({'word': [], 'definition': [], 'example': []}).to_csv(vocab_csv_path, index=False)

def fetch_wikipedia_paragraph(topic, paragraph_number=1):
    """
    Fetch a specific paragraph of a Wikipedia article for the given topic.
    :param topic: The topic to search on Wikipedia.
    :param paragraph_number: Which paragraph to fetch (1 for the first, 2 for the second).
    """
    try:
        # Format the topic into a Wikipedia search URL
        url = f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}"
        response = requests.get(url)
        
        if response.status_code != 200:
            return f"Sorry, I couldn't fetch details about {topic} from Wikipedia."

        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find all paragraphs in the main content
        paragraphs = soup.find_all('p')
        
        # Filter out empty or irrelevant paragraphs with regex
        content_paragraphs = [
            re.sub(r'\[\d+\]', '', p.text.strip())  # Remove [number] references
            for p in paragraphs
            if p.text.strip()
        ]
        
        # Return the requested paragraph
        if len(content_paragraphs) >= paragraph_number:
            return content_paragraphs[paragraph_number - 1]
        else:
            return f"Sorry, I couldn't find more information about {topic} on Wikipedia."
    except Exception as e:
        logging.error(f"Error fetching Wikipedia content for {topic}: {e}")
        return "Sorry, an error occurred while fetching additional information."
    
def clean_csv(file_path, required_columns):
    try:
        df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
        df.drop_duplicates(inplace=True)
        if not set(required_columns).issubset(df.columns):
            logging.error(f"CSV {file_path} is missing required columns: {required_columns}")
            return None
        df.to_csv(file_path, index=False, encoding='utf-8')
        logging.info(f"CSV {file_path} cleaned successfully.")
        return df
    except Exception as e:
        logging.error(f"Error cleaning CSV {file_path}: {e}")
        return None

 #Cleann the vocabulary and responses CSV at startup
 
vocab_df = clean_csv(vocab_csv_path, ['word', 'definition', 'example'])
response_df = clean_csv(response_csv_path, ['keyword', 'response'])

# Function to extract key terms using NLP
def extract_key_terms(user_input):
    entities = nlp(user_input)
    key_terms = [entity['word'] for entity in entities if entity['score'] > 0.5]
    return " ".join(key_terms) if key_terms else user_input

# Function to provide pronunciation using phonemes
def get_pronunciation(word):
    phonemes = g2p(word)
    return " ".join(phonemes)

def track_conversation(user_input, bot_response):
    """
    Tracks user input and bot response, maintaining a history of up to 5 exchanges.
    """
    conversation_history.append({'user': user_input, 'bot': bot_response})
    if len(conversation_history) > 5:  # Limit memory length
        conversation_history.pop(0)

# Function to clean and rewrite CSV


# Global dataframes
terms_df = pd.DataFrame()
vocab_df = pd.DataFrame()
response_df = pd.DataFrame()

# Path to your CSV files
vocab_csv_path = 'static/vocabulary.csv'
response_csv_path = 'static/responses.csv'


# Initialize rocket_data and terms_df globally
rocket_data = pd.DataFrame()
terms_df = pd.DataFrame()

# Load NLP pipeline for extracting key terms from user input
nlp = pipeline("ner", model="dslim/bert-base-NER", tokenizer="dslim/bert-base-NER")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Path to your CSV file
csv_file_path = 'static/science_terms.csv'

# Function to clean and rewrite the CSV
def clean_and_rewrite_csv(file_path):
    try:
        # Load the CSV with appropriate error handling for encoding issues
        df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')

        # Drop any duplicate rows
        df.drop_duplicates(inplace=True)

        # Ensure correct column headers
        if not {'keyword', 'description'}.issubset(df.columns):
            logging.error("The CSV file is missing required columns. Adding missing columns.")
            for column in ['keyword', 'description']:
                if column not in df.columns:
                    df[column] = ""

        # Reorder columns to ensure 'keyword' and 'description' are the first two columns
        columns = ['keyword', 'description'] + [col for col in df.columns if col not in ['keyword', 'description']]
        df = df[columns]

        # Save the cleaned dataframe back to the CSV file
        df.to_csv(file_path, index=False, encoding='utf-8')
        logging.info("CSV file has been cleaned and saved successfully.")

    except Exception as e:
        logging.error(f"Error occurred while cleaning the CSV file: {e}")

# Function to clean and rewrite the CSV
def clean_and_rewrite_csv(file_path):
    try:
        # Load the CSV with appropriate error handling for encoding issues
        df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')

        # Drop any duplicate rows
        df.drop_duplicates(inplace=True)

        # Ensure correct column headers
        if not {'keyword', 'description'}.issubset(df.columns):
            logging.error("The CSV file is missing required columns. Adding missing columns.")
            for column in ['keyword', 'description']:
                if column not in df.columns:
                    df[column] = ""

        # Reorder columns to ensure 'keyword' and 'description' are the first two columns
        columns = ['keyword', 'description'] + [col for col in df.columns if col not in ['keyword', 'description']]
        df = df[columns]

        # Save the cleaned dataframe back to the CSV file
        df.to_csv(file_path, index=False, encoding='utf-8')
        logging.info("CSV file has been cleaned and saved successfully.")

    except Exception as e:
        logging.error(f"Error occurred while cleaning the CSV file: {e}")


# Flask app setup
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Check predefined responses


from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Load SciBERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
model = AutoModelForTokenClassification.from_pretrained("allenai/scibert_scivocab_uncased")

# Create an NLP pipeline for NER
scibert_nlp_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

def scibert_nlp(user_input):
    try:
        # Apply the NER pipeline
        ner_results = scibert_nlp_pipeline(user_input)
        # Format results into a simplified structure
        entities = [{'word': result['word'], 'entity': result['entity']} for result in ner_results]
        return entities
    except Exception as e:
        logging.error(f"Error in SciBERT NLP: {e}")
        return []

terms_df = load_terms_df()
rocket_data = pd.DataFrame()

def get_response(user_input):
    global current_topic
    global terms_df
    user_input_normalized = user_input.strip().lower()

    if conversation_history:
        if any(x in user_input.lower() for x in ["it", "this", "that"]):
            for entry in reversed(conversation_history):
                if entry['user'] not in ["hello", "hi"]:
                    user_input = f"{entry['user']}. {user_input}"
                    break

    ambiguous_phrases = ["it", "this", "that", "tell me more", "explain more", "go on", "undertand"]
    if any(phrase in user_input_normalized for phrase in ambiguous_phrases):
        if current_topic:
            response_text =  clean_text_for_audio(fetch_wikipedia_paragraph(current_topic))
            audio_filename = f"static/audio_{int(time.time())}.mp3"
            save_audio_gtts(response_text, audio_filename)
            return {'response': response_text, 'audio_path': audio_filename}
        else:
            return {'response': "Could you clarify what you’d like to know more about?", 'audio_path': None}

    negative_responses = ["stupid", "dumb", "bitch", "idiot", "fuck", "fuck you", "useless", "garbage", "hate", "hate you", "shut up", "bitch", "dummy", "crazy", "monkey", "weirdo", "dumb blonde", "blonde"]
    for word in negative_responses:
        if word in user_input.lower():
                    response_text="I'm here to help, and I'm always learning! Let's keep things respectful."
                    audio_filename = f"static/audio_{int(time.time())}.mp3"
                    save_audio_gtts(response_text, audio_filename)
                    return {'response': response_text, 'audio_path': audio_filename}

    if "sorry" in user_input.lower():
                    response_text="No worries! Let's continue with learning about STEM topics."
                    audio_filename = f"static/audio_{int(time.time())}.mp3"
                    save_audio_gtts(response_text, audio_filename)
                    return {'response': response_text, 'audio_path': audio_filename}


    key_terms = extract_key_term_nlp(user_input_normalized)
    if key_terms:
        user_input_normalized = key_terms

  # Extract the key terms from user input using the NLP model
    key_terms = extract_key_term_nlp(user_input_normalized)
    if key_terms:
        user_input_normalized = key_terms
    
    # Handle "water rocket" or "rocket" queries
    if "water rocket" in user_input_normalized or "rocket" in user_input_normalized:
        # Ensure rocket_data.csv is loaded
        global rocket_data
        if rocket_data.empty:
            try:
                rocket_data = pd.read_csv('static/rocket.csv', encoding='utf-8', on_bad_lines='skip')
                logging.info("Rocket data CSV loaded successfully.")
                logging.info(f"First 10 rows of Rocket Data CSV content: \n{rocket_data.head(10)}")
            except FileNotFoundError:
                logging.error("Rocket data CSV file not found.")
                return "I'm unable to provide details on water rockets at the moment because the data file is missing. Please check back later."
            except Exception as e:
                logging.error(f"Error loading rocket data CSV: {e}")
                return "I'm unable to provide details on water rockets at the moment. Please try again later."

        # Extract details from the rocket_data
        water_rocket_steps = rocket_data[rocket_data['Experiment'].str.lower() == "water rocket"]
        response_steps = []

        for _, row in water_rocket_steps.iterrows():
            if pd.notna(row['Step']):
                step_text = f"<br><strong>Step {int(row['Step'])}</strong>: {row['Details']}"
                if pd.notna(row['Notes']):
                    step_text += f" <br><strong style='color:red;'><br>Remember!</strong> {row['Notes']}"
                if pd.notna(row['ImagePath']):
                    step_text += f" <br><img width='200px' src= '{row['ImagePath']}'>"
                response_steps.append(step_text + "<br>")  # Adding <br> tag after each step for better formatting

        steps_text = " ".join(response_steps)
        clean_steps_text = re.sub(r'<[^>]+>', '', steps_text)  # Remove HTML tags for audio

        # Generate audio without HTML tags
        audio_filename = f"static/audio_{int(time.time())}.mp3"
        save_audio_gtts(clean_steps_text, audio_filename)

        return {'response': steps_text, 'audio_path': audio_filename}

    # Handle vocabulary terms from science_terms.csv

    if terms_df is not None and not terms_df.empty and 'keyword' in terms_df.columns:
        matched_term = None
        highest_score = 0
        for keyword in terms_df['keyword'].str.lower():
            score = fuzz.partial_ratio(user_input_normalized, keyword)
            if score > highest_score and score > 70:
                highest_score = score
                matched_term = keyword
                current_topic = keyword

        if matched_term:
            response_text =  clean_text_for_audio(terms_df[terms_df['keyword'].str.lower() == matched_term]['description'].iloc[0])
            audio_filename = f"static/audio_{int(time.time())}.mp3"
            save_audio_gtts(response_text, audio_filename)
            return {'response': response_text, 'audio_path': audio_filename}

    # GPT-2 fallback
    # Note: We use user_input, not user_input_normalized, because user_input is the original string.
    request_user_input = request.form.get('user_input', '')
    if not request_user_input.strip():
        request_user_input = user_input  # If form data not found, fallback to user_input

    entities = scibert_nlp(request_user_input)
    input_ids = gpt2_tokenizer.encode(request_user_input, return_tensors='pt')
    outputs = gpt2_model.generate(
        input_ids, 
        max_length=100, 
        num_return_sequences=1, 
        no_repeat_ngram_size=2, 
        do_sample=True, 
        top_p=0.95, 
        top_k=50
    )
    generated_text = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
    audio_filename = f"static/audio_{int(time.time())}.mp3"
    save_audio_gtts(generated_text, audio_filename)

    return {'response': generated_text, 'audio_path': audio_filename}


@app.route('/learn_word', methods=['POST'])
def learn_word():
    user_input = request.form.get('word', '').strip().lower()
    if response_df is not None and not response_df.empty and 'keyword' in response_df.columns:
        matched_response = response_df[response_df['keyword'].str.lower() == user_input]
        if not matched_response.empty:
            response = matched_response.iloc[0]['response']
            audio_filename = f"static/audio_{int(time.time())}.mp3"
            save_audio_gtts(response, audio_filename)
            return jsonify({'response': clean_text_for_audio(response_text), 'audio_path': audio_path})

    if vocab_df is not None and not vocab_df.empty and 'word' in vocab_df.columns:
        matched_row = vocab_df[vocab_df['word'].str.lower() == user_input]
        if not matched_row.empty:
            word = matched_row.iloc[0]['word']
            definition = matched_row.iloc[0]['definition']
            example = matched_row.iloc[0]['example']
            pronunciation = get_pronunciation(word)
            global current_topic
            current_topic = word
            response_text =  clean_text_for_audio(f"{word}: {pronunciation}. Definition: {definition}. Example: {example}.")
            audio_filename = f"static/audio_{word}.mp3"
            save_audio_gtts(response_text, audio_filename)
            return jsonify({'word': word, 'definition': definition, 'example': example, 'pronunciation': pronunciation, 'audio_path': audio_filename})

    return jsonify({'error': "Word not found in vocabulary."})          



@app.route('/get_definition', methods=['POST'])
def get_definition_response():
    user_input = request.form.get('user_input', '').strip()
    response_data = get_response(user_input)

    if isinstance(response_data, dict):
        response_text = response_data['response']
        audio_path = response_data['audio_path']
    else:
        response_text = response_data
        audio_path = None
    track_conversation(user_input, response_text)
    return jsonify({'response': response_text, 'audio_path': audio_path})



@app.route('/conversation_history', methods=['GET'])
def get_conversation_history():
    """
    Returns the current conversation history as JSON.
    """
    return jsonify({'history': conversation_history})

@app.route('/')
def index():
    return render_template('index.html')

def clean_csv(file_path, required_columns):
    try:
        df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
        df.drop_duplicates(inplace=True)
        if not set(required_columns).issubset(df.columns):
            logging.error(f"CSV {file_path} is missing required columns: {required_columns}")
            return None
        df.to_csv(file_path, index=False, encoding='utf-8')
        logging.info(f"CSV {file_path} cleaned successfully.")
        return df
    except Exception as e:
        logging.error(f"Error cleaning CSV {file_path}: {e}")
        return None

# Clean the vocabulary and responses CSV at startup
vocab_df = clean_csv(vocab_csv_path, ['word', 'definition', 'example'])
response_df = clean_csv(response_csv_path, ['keyword', 'response'])



# Function to extract key terms using NLP
def extract_key_terms(user_input):
    entities = nlp(user_input)
    key_terms = [entity['word'] for entity in entities if entity['score'] > 0.5]

    return " ".join(key_terms) if key_terms else user_input
    
# Function to provide pronunciation using phonemes
def get_pronunciation(word):
    phonemes = g2p(word)
    return " ".join(phonemes)



def save_audio_gtts(text, filename):
    try:
        tts = gTTS(text=text, lang='en')
        tts.save(filename)
        logging.info(f"Audio saved to {filename} using gTTS.")
    except Exception as e:
        logging.error(f"Error saving audio: {e}")

def clean_text_for_audio(text):
    """Remove HTML tags from text."""
    return re.sub(r'<[^>]+>', '', text)





@app.route('/book-suggestions', methods=['GET'])
def book_suggestions():
    books = get_kids_science_books()
    kits = get_science_kits()
    return render_template('book-suggestions.html', books=books, kits=kits)



def get_science_kits():
    """
    Fetches science kits using the eBay Browse API.
    """
    api_url = "https://api.ebay.com/buy/browse/v1/item_summary/search"
    headers = {
        "Authorization": "Bearer YOUR_EBAY_API_KEY",  # Replace with your eBay API key
        "Content-Type": "application/json",
    }
    params = {
        "q": "science kits for kids",  # Search query
        "limit": 10,  # Limit the number of results
    }
    response = requests.get(api_url, headers=headers, params=params)

    if response.status_code != 200:
        print(f"Failed to fetch kits. Status code: {response.status_code}")
        return []

    data = response.json()
    kits = []
    for item in data.get("itemSummaries", []):
        kits.append({
            "title": item.get("title", "No Title"),
            "price": item.get("price", {}).get("value", "N/A"),
            "link": item.get("itemWebUrl", "#"),
            "image": item.get("image", {}).get("imageUrl", ""),
        })

    return kits

def get_kids_science_books():
    """
    Fetches kids' science books using the Google Books API.
    """
    api_url = "https://www.googleapis.com/books/v1/volumes"
    params = {
        "q": "kids science books",  # Search query
        "maxResults": 10,  # Limit the number of results
    }
    response = requests.get(api_url, params=params)

    if response.status_code != 200:
        print(f"Failed to fetch books. Status code: {response.status_code}")
        return []

    data = response.json()
    books = []
    for item in data.get("items", []):
        book_info = item.get("volumeInfo", {})
        books.append({
            "title": book_info.get("title", "No Title"),
            "price": "N/A (Books API does not provide prices)",
            "link": book_info.get("infoLink", "#"),
            "image": book_info.get("imageLinks", {}).get("thumbnail", ""),
        })

    return books


@app.route('/science-projects')
def show_science_projects():
    # Render the previously generated HTML page
    return render_template('science-projects.html')


# URL of the target page
url = "https://www.education.com/science-fair/science/"

# Send a GET request to fetch the page content
response = requests.get(url)
html_content = response.text

# Parse HTML using BeautifulSoup
soup = BeautifulSoup(html_content, "html.parser")

# Locate the container holding the science projects
# Inspect the page to find a unique class or ID wrapping all projects
# According to provided HTML snippet, projects are within 'div.search-module_results_2Lx3Z' or '.SearchResults-module_list_2AV2c'
projects_container = soup.select_one('.SearchResults-module_list_2AV2c')

if not projects_container:
    print("Could not find the projects container. The page might be dynamically loaded via JS.")
    # Consider using Selenium if this fails
    exit()

projects = projects_container.select('.SearchResults-module_result_3-uXL.science-fair')

# Store scraped data
data = []
for proj in projects:
    # Each project link and title
    link_tag = proj.select_one('a.SearchResults-module_link_3VC4l')
    if not link_tag:
        continue

    project_url = link_tag.get('href')
    project_title = link_tag.select_one('.Title-module_title_21JAg span')
    project_title = project_title.get_text(strip=True) if project_title else "No Title"

    # Thumbnail or image
    image_tag = proj.select_one('.ContentCard-module_image_1D6-7 img, .ContentCard-module_image_1D6-7 svg')
    # Check if it's an img or fallback to SVG if no img
    if image_tag and image_tag.name == 'img':
        image_url = image_tag.get('data-src') or image_tag.get('src')
    else:
        # If it's an SVG or no image at all
        image_url = None

    # Description
    description_tag = proj.select_one('.ContentCard-module_description_1ejK0')
    description = description_tag.get_text(strip=True) if description_tag else "No description"

    # Store result
    data.append({
        'title': project_title,
        'url': project_url,
        'image': image_url,
        'description': description
    })

# Now we have the data extracted, we can reconstruct the format in `science-projects.html`.
# For example, writing a simple HTML structure with the extracted data:
html_output = ['<html><head><title>Science Projects</title></head><body><h1>Science Projects</h1><div class="projects">']

for item in data:
    html_output.append('<div class="project">')
    html_output.append(f'<h2><a href="{item["url"]}">{item["title"]}</a></h2>')
    if item['image']:
        html_output.append(f'<img src="{item["image"]}" alt="{item["title"]}" />')
    html_output.append(f'<p>{item["description"]}</p>')
    html_output.append('</div>')

html_output.append('</div></body></html>')

# Write to science-projects.html
with open('science-projects.html', 'w', encoding='utf-8') as f:
    f.write("\n".join(html_output))

print("Scraping complete. Check science-projects.html for the output.")



if __name__ == '__main__':
    app.run(debug=True)
    while True:
        logging.info("Starting CSV cleanup...")
        clean_and_rewrite_csv(csv_file_path)

