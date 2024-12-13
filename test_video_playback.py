import cv2

# Path to the video file you want to test
video_path = "static/output_video.mp4"

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print(f"Error: Unable to open video file {video_path}")
    exit()

# Read and display video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or no frames read.")
        break
    cv2.imshow("Test Video", frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release resources and close windows
cap.release()
cv2.destroyAllWindows()
