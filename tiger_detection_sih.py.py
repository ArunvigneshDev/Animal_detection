import cv2
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from playsound import playsound

# Load the pre-trained MobileNetV2 model
model = MobileNetV2(weights='imagenet')

# Path to the gunshot sound
gunshot_sound_path = "C:\Project farm land\gunshot.mp3"

# Function to preprocess the frame for model prediction
def preprocess_frame(frame):
    img = cv2.resize(frame, (224, 224))  # Resize to match the model's input size
    img = image.img_to_array(img)  # Convert to an array
    img = np.expand_dims(img, axis=0)  # Expand dimensions to match the model input
    img = preprocess_input(img)  # Preprocess using MobileNetV2 function
    return img

# Function to detect tiger and play gunshot sound
def detect_tiger(frame):
    processed_frame = preprocess_frame(frame)
    preds = model.predict(processed_frame)
    decoded_preds = decode_predictions(preds, top=1)[0]
    
    # Check if the detected object is a tiger (category ID for tiger is n02129604 in ImageNet)
    if decoded_preds[0][0] == 'n02129604':  # Tiger detected
        print("Tiger detected! Playing gunshot sound.")
        playsound(gunshot_sound_path)

# Open laptop camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect tiger in the current frame
    detect_tiger(frame)

    # Display the frame
    cv2.imshow('Tiger Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and close windows
cap.release()
cv2.destroyAllWindows()
