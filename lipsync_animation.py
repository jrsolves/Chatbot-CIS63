import pygame  # For audio playback and displaying animation
import cv2  # OpenCV for video writing
from gtts import gTTS  # Google Text-to-Speech
import time
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"  # This sets pygame to use a dummy video driver
os.environ["SDL_AUDIODRIVER"] = "dummy"  # Run pygame without initializing audio

import subprocess  # For running ffmpeg

# Function to save speech using gTTS
def save_audio(text, filename="static/output_audio.mp3", lang="en"):
    tts = gTTS(text=text, lang=lang, slow=False)
    tts.save(filename)

# Generate the audio file
save_audio("Hello, this is a lipsync animation test.")

# Initialize pygame and the mixer for playing audio
pygame.init()
pygame.mixer.init()

# Load the audio file
audio_path = "static/output_audio.mp3"
if not os.path.exists(audio_path):
    raise FileNotFoundError("The audio file was not found!")
audio = pygame.mixer.Sound(audio_path)

# Create a simple window for the animation
screen_width, screen_height = 800, 600
screen = pygame.display.set_mode((screen_width, screen_height))

pygame.display.set_caption("Lipsync Animation")

# Scaling factor for images
scale_factor = 0.5  # Adjust this to change the size of all images proportionally

# Load and scale character image
character_image = pygame.image.load("static/assets/character.png")
character_image = pygame.transform.smoothscale(character_image, 
    (int(character_image.get_width() * scale_factor), int(character_image.get_height() * scale_factor)))
character_image = pygame.transform.flip(character_image, True, False)  # Flip once horizontally
character_rect = character_image.get_rect(center=(400, 300))

# Load, scale, and flip the mouth images horizontally
mouth_open = pygame.image.load("static/assets/mouth_open.png")
mouth_open = pygame.transform.smoothscale(mouth_open,
    (int(mouth_open.get_width() * 0.5), int(mouth_open.get_height() * 0.5)))
mouth_open = pygame.transform.flip(mouth_open, True, False)  # Flip once horizontally

mouth_closed = pygame.image.load("static/assets/mouth_closed.png")
mouth_closed = pygame.transform.smoothscale(mouth_closed,
    (int(mouth_closed.get_width() * 0.5), int(mouth_closed.get_height() * 0.5)))
mouth_closed = pygame.transform.flip(mouth_closed, True, False)  # Flip once horizontally

# Get rects for the mouth images
mouth_open_rect = mouth_open.get_rect(center=character_rect.center)
mouth_closed_rect = mouth_closed.get_rect(center=character_rect.center)






# Prepare OpenCV video writer
video_filename = "static/output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Try 'XVID' if 'mp4v' doesn't work
fps = 30  # Frames per second
out = cv2.VideoWriter(video_filename, fourcc, fps, (screen_width, screen_height))

if not out.isOpened():
    raise ValueError("Failed to open VideoWriter. Check codec and path.")

# Check if the VideoWriter was successfully initialized
if not out.isOpened():
    raise ValueError("Failed to open VideoWriter. Check codec and file path.")

# Basic variables for animation
running = True
clock = pygame.time.Clock()

# Play the audio
audio.play()
audio_length = audio.get_length()
start_time = time.time()

# Animation loop
# Animation loop
while running:
    screen.fill((255, 255, 255))  # White background
    screen.blit(character_image, character_rect)

    current_time = time.time() - start_time
    if current_time > audio_length:
        running = False

    if int(current_time * 10) % 2 == 0:
        screen.blit(mouth_open, mouth_open_rect)
    else:
        screen.blit(mouth_closed, mouth_closed_rect)

    pygame.display.flip()  # Update the display

    frame = pygame.surfarray.array3d(pygame.display.get_surface())
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Rotate the frame by 90 degrees clockwise
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    # Debugging: Check if the frame is valid
    if frame is None or frame.size == 0:
        print("Error: Captured frame is empty.")
    else:
        print("Frame captured successfully.")


    # Display the frame for debugging purposes
    #cv2.imshow("Current Frame", frame)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break
    #frame = cv2.flip(frame, 1)  # Flip the frame horizontally with OpenCV
    out.write(frame)

    clock.tick(fps)
pygame.quit()
out.release()
cv2.destroyAllWindows()


# Wait briefly before closing to analyze the output
#pygame.time.delay(3000)  # Wait for 3 seconds

# Clean up pygame
#pygame.quit()
#out.release()
#cv2.destroyAllWindows()

# Define the path to ffmpeg executable
ffmpeg_path = r"C:\ffmpeg\bin\ffmpeg.exe"  # Ensure this is the correct path to ffmpeg

# Define the name for the converted video
converted_video_filename = "static/output_converted_video.mp4"

# Combine video with audio using ffmpeg
subprocess.run([
    ffmpeg_path, "-y",  # Overwrite without asking
    "-i", video_filename,
    "-i", audio_path,
    "-c:v", "libx264",
    "-c:a", "aac",
    "-strict", "experimental",
    "-shortest",
    converted_video_filename
], check=True)

print(f"Converted video created successfully: {converted_video_filename}")
