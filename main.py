import cv2
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np

# Load the MTCNN face detection model
detection_model = MTCNN()

# Load your trained emotion classification model
classify_model = load_model('models/vgg16.h5')

video_name = input("Enter Video File Name:")

# Define the video directory and the classes
video_dir = f'saved_videos/{video_name}.avi'
emotion_labels = ['Confidence', 'Confusion', 'Nervousness']

# Define a function for preprocessing face images
def preprocess_image(img):
    img = cv2.resize(img, (96, 96))
    img = preprocess_input(img)
    return img

# Define a function to classify emotions from face images
def classify_emotion(img):
    # Preprocess the image
    img = preprocess_image(img)
    # Make predictions using the classification model
    predictions = classify_model.predict(np.expand_dims(img, axis=0))[0]
    print(predictions)
    # Get the predicted accuracies
    accuracies = [accuracy for accuracy in predictions]
    # Sort the emotion labels and accuracies based on accuracies in descending order
    emotion_labels_sorted = [label for _, label in sorted(zip(accuracies, emotion_labels), reverse=True)]
    accuracies_sorted = sorted(accuracies, reverse=True)
    return emotion_labels_sorted, accuracies_sorted

# Define the codec for video recording
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# Open a video capture object
cap = cv2.VideoCapture(0)
# Get the width and height of the video frame
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a VideoWriter object to save the video
out = cv2.VideoWriter(video_dir, fourcc, 10.0, (frame_width, frame_height))

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform face detection using MTCNN
    faces = detection_model.detect_faces(frame)
    for face in faces:
        x, y, w, h = face['box']
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # Extract the face region
        face_img = frame[y:y+h, x:x+w]
        # Classify the emotions from the face image
        emotion_labels, accuracies = classify_emotion(face_img)
        # Display the emotion labels and accuracies on the frame
        for i, (emotion_label, accuracy) in enumerate(zip(emotion_labels, accuracies)):
            text = f"{emotion_label} ({accuracy:.2f})"
            cv2.putText(frame, text, (x, y + 30 * i - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
    
    # Write the frame to the output video
    out.write(frame)

    # Display the resulting frame
    frame = cv2.resize(frame, (921, 600))
    cv2.imshow('Emotion Recognition', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture object and close all windows
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Video saved to {video_dir}")