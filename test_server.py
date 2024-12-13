import os
import time
import logging
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import pygame
import cv2
import subprocess
from gtts import gTTS
import spacy

# Initialize spaCy and Flask
nlp = spacy.load("en_core_web_sm")
app = Flask(__name__)

# Load CSV files
definitions_df = pd.read_csv('static/science_terms.csv')
responses_df = pd.read_csv('static/response.csv')

# Logging configuration
logging.basicConfig(level=logging.INFO)

def save_audio_gtts(text, filename, retries=3, delay=5):
    """Save audio with retries to handle temporary issues."""
    for attempt in range(retries):
        try:
            tts = gTTS(text=text, lang='en')
            tts.save(filename)
            logging.info(f"Audio saved as {filename}")
            break
        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(delay)

def create_thinking_audio():
    """Generate audio saying 'That's a very good question, let me think about that.'"""
    audio_path = "static/thinking_audio.mp3"
    if not os.path.exists(audio_path):
        save_audio_gtts("That's a very good question, let me think about that.", audio_path)
    return audio_path

def create_animation_video(video_filename, audio_path):
    """Create animation video using Pygame and OpenCV for lipsync."""
    pygame.init()
    screen_width, screen_height = 800, 600
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Lipsync Animation")

    character_image = pygame.image.load("static/assets/character.png")
    character_image = pygame.transform.smoothscale(character_image, 
        (int(character_image.get_width() * 0.5), int(character_image.get_height() * 0.5)))
    character_rect = character_image.get_rect(center=(400, 300))

    mouth_open = pygame.image.load("static/assets/mouth_open.png")
    mouth_open = pygame.transform.smoothscale(mouth_open, 
        (int(mouth_open.get_width() * 0.5), int(mouth_open.get_height() * 0.5)))
    mouth_closed = pygame.image.load("static/assets/mouth_closed.png")
    mouth_closed = pygame.transform.smoothscale(mouth_closed, 
        (int(mouth_closed.get_width() * 0.5), int(mouth_closed.get_height() * 0.5)))

    mouth_open_rect = mouth_open.get_rect(center=character_rect.center)
    mouth_closed_rect = mouth_closed.get_rect(center=character_rect.center)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30
    out = cv2.VideoWriter(video_filename, fourcc, fps, (screen_width, screen_height))

    try:
        pygame.mixer.init()
        pygame.mixer.music.load(audio_path)
        audio_length = pygame.mixer.Sound(audio_path).get_length()

        running = True
        clock = pygame.time.Clock()
        start_time = time.time()

        while running:
            screen.fill((255, 255, 255))
            screen.blit(character_image, character_rect)

            current_time = time.time() - start_time
            if current_time > audio_length:
                running = False

            if int(current_time * 10) % 2 == 0:
                screen.blit(mouth_open, mouth_open_rect)
            else:
                screen.blit(mouth_closed, mouth_closed_rect)

            frame = pygame.surfarray.array3d(pygame.display.get_surface())
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            frame = cv2.flip(frame, 1)
            out.write(frame)

            clock.tick(fps)

        # Ensure the final frame has the mouth closed
        screen.fill((255, 255, 255))
        screen.blit(character_image, character_rect)
        screen.blit(mouth_closed, mouth_closed_rect)
        frame = pygame.surfarray.array3d(pygame.display.get_surface())
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        frame = cv2.flip(frame, 1)
        out.write(frame)

    finally:
        out.release()
        cv2.destroyAllWindows()
        pygame.quit()

def get_response(user_input):
    """Retrieve response from CSV based on user input."""
    doc = nlp(user_input.lower())
    term = ' '.join([token.text for token in doc if not token.is_stop and token.is_alpha])

    science_result = definitions_df[definitions_df['keyword'].str.lower() == term.lower()]
    if not science_result.empty:
        return science_result['description'].iloc[0]

    general_result = responses_df[responses_df['keyword'].str.lower() == term.lower()]
    if not general_result.empty:
        return general_result['response'].iloc[0]

    return "I'm not sure how to respond to that. Can you ask me something else?"

def generate_unique_filename(prefix="output"):
    timestamp = int(time.time())
    return f"static/{prefix}_{timestamp}.mp4"

def convert_video(video_filename, audio_path, output_filename):
    """Combine video and audio using FFmpeg."""
    ffmpeg_path = r"C:\ffmpeg\bin\ffmpeg.exe"
    try:
        subprocess.run([
            ffmpeg_path, "-y", "-i", video_filename, "-i", audio_path,
            "-c:v", "libx264", "-c:a", "aac", "-strict", "experimental",
            "-shortest", output_filename
        ], check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error during video conversion: {e}")
        raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_definition', methods=['POST'])
def get_definition_response():
    user_input = request.form['user_input']
    response_text = get_response(user_input)
    word_count = len(response_text.split())

    if word_count > 200:
        thinking_audio = create_thinking_audio()
        video_filename = generate_unique_filename("output")
        converted_video_filename = generate_unique_filename("converted_output")

        # Generate thinking response
        save_audio_gtts("That's a very good question, let me think about that.", thinking_audio)
        
        # Generate final audio and video files
        save_audio_gtts(response_text, f"static/audio_{int(time.time())}.mp3")
        create_animation_video(video_filename, thinking_audio)
        
        # Convert to video format
        convert_video(video_filename, thinking_audio, converted_video_filename)
        
        # Send JSON response
        return jsonify({
            'response': response_text,
            'thinking_audio': thinking_audio,
            'show_thinking': True,
            'video_path': converted_video_filename
        })

    else:
        svg_animation_path = "static/animation.svg"
        timestamp = int(time.time())
        audio_filename = f"static/audio_{timestamp}.mp3"
        save_audio_gtts(response_text, audio_filename)

        return jsonify({'response': response_text, 'svg_path': svg_animation_path, 'audio_path': audio_filename})

@app.route('/get_video/<filename>')
def get_video(filename):
    video_path = os.path.join("static", filename)
    if os.path.exists(video_path):
        return send_file(video_path, as_attachment=False)
    else:
        return "Video not found", 404

def run_cleanup_script():
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cleanup.bat')
    if os.path.exists(script_path):
        subprocess.run([script_path], shell=True, check=True)

if __name__ == '__main__':
    run_cleanup_script()
    app.run(debug=True)
