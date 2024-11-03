from flask import Flask, render_template, request, jsonify
import pandas as pd
import cv2
from gtts import gTTS
from moviepy.editor import *
from phonemizer import phonemize
from pydub import AudioSegment
import requests
from bs4 import BeautifulSoup
import os

app = Flask(__name__)

# Load science terms and project keywords from CSVs
def load_csv_definitions(file_path):
    df = pd.read_csv(file_path)
    return dict(zip(df["keyword"].str.lower(), df["description"])), dict(zip(df["keyword"].str.lower(), df["image_path"]))

# Load both science terms and science projects
science_terms, science_terms_images = load_csv_definitions("science_terms.csv")
science_projects, project_images = load_csv_definitions("science_projects.csv")

# Check both dictionaries for a keyword
def get_explanation(term):
    term = term.lower()
    if term in science_terms:
        return science_terms[term], science_terms_images[term]
    elif term in science_projects:
        return science_projects[term], project_images[term]
    else:
        return scrub_web_for_definition(term), None

# Scrub the web for definitions if keyword is not found
def scrub_web_for_definition(term):
    headers = {"User-Agent": "Mozilla/5.0"}
    websites = ["https://www.dictionary.com", "https://www.thoughtco.com"]
    for site in websites:
        response = requests.get(f"{site}/search?q={term}", headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            definition = soup.find("p").get_text()
            if definition:
                return definition
    return "Definition not found in CSV or on the websites."

# Detect relevant keywords from text and return associated images
def detect_keywords(text, images_dict):
    detected_images = []
    for keyword, image_path in images_dict.items():
        if keyword in text.lower():
            detected_images.append(image_path)
    return detected_images

# Generate audio for response
def generate_audio(text, audio_file="static/response.mp3"):
    tts = gTTS(text)
    tts.save(audio_file)

# Generate video with keyword images
def generate_video(text, audio_file="static/response.mp3", output_video="static/response.mp4"):
    keyword_images = {**science_terms_images, **project_images}
    images_to_show = detect_keywords(text, keyword_images)
    
    avatar_image = cv2.imread("path/to/avatar_image.jpg")
    avatar_with_mouth = avatar_image.copy()
    mouth_imgs = {
        "a": cv2.imread("path/to/a_image.png"),
        "e": cv2.imread("path/to/e_image.png")
    }
    
    audio_duration = len(AudioSegment.from_mp3(audio_file)) / 1000.0
    fps = 30
    frame_width, frame_height = avatar_image.shape[1], avatar_image.shape[0]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

    total_frames = int(fps * audio_duration)
    inserted_images_frames = total_frames // (len(images_to_show) + 1)
    current_frame = 0

    for image_path in images_to_show:
        keyword_image = cv2.imread(image_path)
        keyword_image = cv2.resize(keyword_image, (frame_width, frame_height))
        for _ in range(int(inserted_images_frames)):
            video.write(keyword_image)
            current_frame += 1

    for phoneme in text_to_phonemes(text):
        mouth_image = mouth_imgs.get(phoneme, mouth_imgs["a"])
        avatar_with_mouth[300:300+mouth_image.shape[0], 250:250+mouth_image.shape[1]] = mouth_image
        for _ in range(int(fps * audio_duration / len(text_to_phonemes(text)))):
            video.write(avatar_with_mouth)
            current_frame += 1
            if current_frame >= total_frames:
                break

    video.release()

# Endpoint for tutorials chatbot
@app.route("/chat_tutorial", methods=["POST"])
def chat_tutorial():
    data = request.get_json()
    user_input = data["user_input"]
    response, image = get_explanation(user_input)
    
    generate_audio(response)
    generate_video(response)

    return jsonify({"response": response, "video_url": "/static/response.mp4"})

if __name__ == "__main__":
    app.run(debug=True)
