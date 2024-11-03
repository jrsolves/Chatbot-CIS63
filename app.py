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
import nltk
from nltk.corpus import cmudict

nltk.download('cmudict')

app = Flask(__name__)

# Load science terms and project keywords from CSVs
# ... (the rest of your existing code remains unchanged) ...

@app.route("/")
def index():
    return render_template("index.html")

# Define the new tutorials route
@app.route("/tutorials")
def tutorials():
    return render_template("tutorials.html")

@app.route("/chat_tutorial", methods=["POST"])
def chat_tutorial():
    data = request.get_json()
    user_input = data["user_input"]
    response, image = get_explanation(user_input)
    
    # Generate the audio and video response based on user input
    generate_audio(response)
    generate_video(response)
    
    # Respond with JSON data containing text response and video URL
    return jsonify({"response": response, "video_url": "/static/response.mp4"})

if __name__ == "__main__":
    app.run(debug=True)
