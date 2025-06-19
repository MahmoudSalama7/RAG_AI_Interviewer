import cv2
import tensorflow as tf
import mediapipe as mp
from deepface import DeepFace
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Add this at the beginning of your script
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Initialize OpenCV Video Capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Emotion labels (order matters for the output)
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'neutral']

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Face Detection
    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

            # Extract the face ROI
            face_roi = frame[y:y+h, x:x+w]

            # Predict emotion using DeepFace
            try:
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False,detector_backend='skip',align=False)
                emotion = result[0]['dominant_emotion']
                emotion_probabilities = result[0]['emotion']

                # Prepare the probabilities in the order of emotion_labels
                probabilities = [emotion_probabilities[label] for label in emotion_labels]

                # Print the output in the desired format
                print(f"{emotion.capitalize()} {probabilities}")

                # Draw the bounding box and emotion label on the frame
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"{emotion}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            except Exception as e:
                print(f"Error in emotion detection: {e}")

    # Display the frame
    cv2.imshow('Emotion Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
