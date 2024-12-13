import sys
import os
import time
import logging
from flask import Flask, render_template, request, jsonify, url_for, send_file
import pandas as pd
from gtts import gTTS
import re
import requests
from bs4 import BeautifulSoup
import random
from g2p_en import G2p
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, pipeline, AutoTokenizer, AutoModelForTokenClassification
from rapidfuzz import fuzz


API_KEY = "your_bing_api_key"
ENDPOINT = "https://api.bing.microsoft.com/v7.0/images/search"

# List of experiment names for which we need images
experiments = [
    "Baking Soda Volcano",
    "Magic Milk",
    "Invisible Ink",
    "Dancing Raisins",
    "Homemade Lava Lamp",
    "Lemon Battery",
    "Rainbow Walking Water",
    "Paper Towel Chromatography",
    "Balloon Rocket",
    "Egg in Vinegar (Naked Egg)",
    "Homemade Slime",
    "Skittles Color Wheel",
    "Cloud in a Jar",
    "Homemade Compass",
    "Mentos and Soda Geyser",
    "Elephant Toothpaste",
    "Static Electricity Balloon",
    "Rainbow Celery",
    "Grow Borax Crystals",
    "Dry Ice Bubble",
    "Water Colors (Biology)",
    "Falling Leaves (Biology)",
    "Hole-y Walls (Biology)",
    "Do Seeds Need Light",
    "Raw Egg Peeler (Chemistry)",
    "Cleaning Pennies (Chemistry)",
    "Balloon Rocket (Physics)",
    "Kids’ Lab Lessons: Why Do Boats Float? (Physics)",
    "Mini Volcano (Earth Science)",
    "Space of Air (Sky Above Us)",
    "Deep Breath (Human Body)",
    "Action-Reaction (Human Body)",

]

# Directory to save images
image_dir = "static/images"
os.makedirs(image_dir, exist_ok=True)

def fetch_image(query, save_path):
    headers = {"Ocp-Apim-Subscription-Key": API_KEY}
    params = {"q": query, "license": "public", "imageType": "photo", "count": 1}

    try:
        response = requests.get(ENDPOINT, headers=headers, params=params)
        response.raise_for_status()
        results = response.json()

        if "value" in results and results["value"]:
            image_url = results["value"][0]["contentUrl"]

            # Download the image
            image_data = requests.get(image_url).content
            with open(save_path, "wb") as image_file:
                image_file.write(image_data)
            print(f"Downloaded: {query} -> {save_path}")
        else:
            print(f"No results found for {query}")

    except Exception as e:
        print(f"Error fetching image for {query}: {e}")

# Loop through the experiments and fetch missing images
for experiment in experiments:
    filename = experiment.replace(" ", "_").replace("(", "").replace(")", "").replace("’", "").replace("-", "_") + ".jpg"
    save_path = os.path.join(image_dir, filename)

    if not os.path.exists(save_path):
        fetch_image(experiment, save_path)
    else:
        print(f"Image already exists: {filename}")


sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

g2p = G2p()

# Define file paths BEFORE loading or using them
csv_file_path = 'static/science_terms.csv'
vocab_csv_path = 'static/vocabulary.csv'
response_csv_path = 'static/responses.csv'

app = Flask(__name__)  # Define the Flask app BEFORE any @app.route decorators

# Load the GPT-2 model and tokenizer
gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("./model_output")
gpt2_model = GPT2LMHeadModel.from_pretrained("./model_output")

# Define the default NER pipeline for DSLIM
nlp = pipeline("ner", model="dslim/bert-base-NER", tokenizer="dslim/bert-base-NER")

# Define SciBERT pipeline
scibert_tokenizer = AutoTokenizer.from_pretrained(
    "allenai/scibert_scivocab_uncased", cache_dir="./cache_dir", trust_remote_code=True
)
scibert_model = AutoModelForTokenClassification.from_pretrained("allenai/scibert_scivocab_uncased")
scibert_nlp_pipeline = pipeline("ner", model=scibert_model, tokenizer=scibert_tokenizer)

conversation_history = []
current_topic = None



def load_experiments():
    try:
        df = pd.read_csv('static/science_projects.csv')
        return df['keyword'].unique()  # Extract unique experiment names
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return []


def save_audio_gtts(text, filename):
    try:
        clean_text = clean_text_for_audio(text)
        tts = gTTS(text=clean_text, lang='en')
        tts.save(filename)
        logging.info(f"Audio saved to {filename} using gTTS.")
    except Exception as e:
        logging.error(f"Error saving audio: {e}")

def clean_text_for_audio(text):
    return re.sub(r'<[^>]+>', '', text)

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

vocab_df = clean_csv(vocab_csv_path, ['word', 'definition', 'example'])
if vocab_df is None:
    vocab_df = pd.DataFrame(columns=['word', 'definition', 'example'])

response_df = clean_csv(response_csv_path, ['keyword', 'response'])
if response_df is None:
    response_df = pd.DataFrame(columns=['keyword', 'response'])

def fetch_wikipedia_paragraph(topic, paragraph_number=1):
    try:
        url = f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}"
        response = requests.get(url)
        if response.status_code != 200:
            return f"Sorry, I couldn't fetch details about {topic} from Wikipedia."
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        content_paragraphs = [
            re.sub(r'\[\d+\]', '', p.text.strip())
            for p in paragraphs if p.text.strip()
        ]
        if len(content_paragraphs) >= paragraph_number:
            return content_paragraphs[paragraph_number - 1]
        else:
            return f"Sorry, I couldn't find more information about {topic} on Wikipedia."
    except Exception as e:
        logging.error(f"Error fetching Wikipedia content for {topic}: {e}")
        return "Sorry, an error occurred while fetching additional information."

def track_conversation(user_input, bot_response):
    conversation_history.append({'user': user_input, 'bot': bot_response})
    if len(conversation_history) > 5:
        conversation_history.pop(0)

def scibert_nlp(user_input):
    try:
        ner_results = scibert_nlp_pipeline(user_input)
        entities = [{'word': result['word'], 'entity': result['entity']} for result in ner_results]
        return entities
    except Exception as e:
        logging.error(f"Error in SciBERT NLP: {e}")
        return []

def extract_key_term_nlp(user_input):
    entities = nlp(user_input)
    key_terms = [entity['word'] for entity in entities if entity['score'] > 0.5 and entity['entity'] in ["MISC", "ORG", "PER", "LOC"]]
    return " ".join(key_terms) if key_terms else user_input

def get_pronunciation(word):
    phonemes = g2p(word)
    return " ".join(phonemes)

# Load terms_df




def load_terms_df():
    try:
        df = pd.read_csv(csv_file_path, encoding='utf-8', on_bad_lines='skip')
        df.drop_duplicates(inplace=True)
        if not {'keyword', 'description'}.issubset(df.columns):
            logging.error("The CSV file is missing required columns 'keyword' and 'description'.")
            return pd.DataFrame(columns=['keyword', 'description'])
        return df
    except FileNotFoundError:
        logging.error("terms CSV file not found.")
        return pd.DataFrame(columns=['keyword', 'description'])
    except Exception as e:
        logging.error(f"Error loading terms CSV: {e}")
        return pd.DataFrame(columns=['keyword', 'description'])

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
    return jsonify({'history': conversation_history})

@app.route('/')
def index():
    return render_template('index.html')

def get_science_kits():
    api_url = "https://api.ebay.com/buy/browse/v1/item_summary/search"
    headers = {
        "Authorization": "Bearer YOUR_EBAY_API_KEY",
        "Content-Type": "application/json",
    }
    params = {
        "q": "science kits for kids",
        "limit": 10,
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
    api_url = "https://www.googleapis.com/books/v1/volumes"
    params = {
        "q": "kids science books",
        "maxResults": 10,
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
            "price": "N/A",
            "link": book_info.get("infoLink", "#"),
            "image": book_info.get("imageLinks", {}).get("thumbnail", ""),
        })
    return books

@app.route('/book-suggestions', methods=['GET'])
def book_suggestions():
    books = get_kids_science_books()
    kits = get_science_kits()
    return render_template('book-suggestions.html', books=books, kits=kits)

@app.route('/science-projects')
def science_projects():
    experiments = load_experiments()
    return render_template('science-projects.html', experiments=experiments)

                 
                           
# Scraping logic
url = "https://www.education.com/science-fair/science/"
response = requests.get(url)
html_content = response.text
soup = BeautifulSoup(html_content, "html.parser")
projects_container = soup.select_one('.SearchResults-module_list_2AV2c')

if projects_container:
    projects = projects_container.select('.SearchResults-module_result_3-uXL.science-fair')
    data = []
    for proj in projects:
        link_tag = proj.select_one('a.SearchResults-module_link_3VC4l')
        if not link_tag:
            continue
        project_url = link_tag.get('href')
        project_title = proj.select_one('.Title-module_title_21JAg span')
        project_title = project_title.get_text(strip=True) if project_title else "No Title"

        image_tag = proj.select_one('.ContentCard-module_image_1D6-7 img, .ContentCard-module_image_1D6-7 svg')
        if image_tag and image_tag.name == 'img':
            image_url = image_tag.get('data-src') or image_tag.get('src')
        else:
            image_url = None

        description_tag = proj.select_one('.ContentCard-module_description_1ejK0')
        description = description_tag.get_text(strip=True) if description_tag else "No description"

        data.append({
            'title': project_title,
            'url': project_url,
            'image': image_url,
            'description': description
        })

    html_output = ['<html><head><title>Science Projects</title></head><body><h1>Science Projects</h1><div class="projects">']
    for item in data:
        html_output.append('<div class="project">')
        html_output.append(f'<h2><a href="{item["url"]}">{item["title"]}</a></h2>')
        if item['image']:
            html_output.append(f'<img src="{item["image"]}" alt="{item["title"]}" />')
        html_output.append(f'<p>{item["description"]}</p>')
        html_output.append('</div>')
    html_output.append('</div></body></html>')

    with open('science-projects.html', 'w', encoding='utf-8') as f:
        f.write("\n".join(html_output))

    print("Scraping complete. Check science-projects.html for the output.")
else:
    print("Could not find the projects container. The page might be dynamically loaded via JS.")



@app.route('/experiment/<experiment>')
def experiment_detail(experiment):
    try:
        # Load the CSV file
        df = pd.read_csv('static/science_projects.csv')

        # Filter rows for the selected experiment
        experiment_name = experiment.replace('_', ' ')
        filtered_df = df[df['keyword'] == experiment_name]

        if filtered_df.empty:
            return "Experiment not found", 404

        # Convert filtered data to a list of dictionaries
        steps = filtered_df.to_dict('records')

        # Render the template with the experiment details
        return render_template('experiment-detail.html', experiment=experiment_name, steps=steps)
    except Exception as e:
        return f"An error occurred: {e}", 500
    

if __name__ == '__main__':
    app.run(debug=True)

    