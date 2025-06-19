from fastapi import FastAPI, File, UploadFile, HTTPException
import cv2
import mediapipe as mp
from deepface import DeepFace
import numpy as np
from typing import List, Dict
import os
import cProfile
import pstats

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Initialize FastAPI
app = FastAPI()

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Emotion labels (order matters for the output)
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'neutral']

def evaluate_overall_mood(results: List[str]) -> Dict[str, float]:
    """
    Evaluate the overall mood based on the aggregated emotion probabilities.
    """
    # Initialize a dictionary to store total probabilities for each emotion
    total_probabilities = {label: 0.0 for label in emotion_labels}
    total_frames = len(results)

    # Handle case where no frames were processed
    if total_frames == 0:
        return {
            "overall_mood": "Undetermined",
            "average_probabilities": {label: 0.0 for label in emotion_labels},
            "message": "No frames were successfully processed."
        }

    # Aggregate probabilities from all frames
    for result in results:
        # Extract the probabilities from the result string
        probabilities = list(map(float, result.split("[")[1].strip("]").split(",")))
        for label, prob in zip(emotion_labels, probabilities):
            total_probabilities[label] += prob

    # Calculate average probabilities
    avg_probabilities = {label: total_probabilities[label] / total_frames for label in emotion_labels}

    # Determine the dominant mood
    dominant_emotion = max(avg_probabilities, key=avg_probabilities.get)
    if dominant_emotion == "happy":
        overall_mood = "Positive"
    elif dominant_emotion == "neutral":
        overall_mood = "Neutral"
    else:
        overall_mood = "Negative"

    return {
        "overall_mood": overall_mood,
        "average_probabilities": avg_probabilities
    }

@app.post("/detect-emotion/")
async def detect_emotion(video: UploadFile = File(...)):
    # Check if the uploaded file is a video
    if not (video.content_type.startswith("video/") or video.filename.endswith(('.mp4', '.mov', '.avi'))):
        raise HTTPException(status_code=400, detail="File must be a video")

    # Save the uploaded video temporarily
    with open("temp_video.mp4", "wb") as buffer:
        buffer.write(video.file.read())

    # Open the video file
    cap = cv2.VideoCapture("temp_video.mp4")
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Failed to open video file")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
    frame_interval = int(fps / 3)  # Process 3 frames per second

    results = []
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process only 3 frames per second
        if frame_id % frame_interval == 0:
            # Convert the frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame with MediaPipe Face Detection
            detection_results = face_detection.process(rgb_frame)

            if detection_results.detections:
                for detection in detection_results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

                    # Extract the face ROI
                    face_roi = frame[y:y+h, x:x+w]

                    # Predict emotion using DeepFace
                    try:
                        # Pass align=False and potentially detector_backend='skip' to avoid internal processing
                        result = DeepFace.analyze(
                            face_roi, 
                            actions=['emotion'], 
                            enforce_detection=False, 
                        )
                        emotion_probabilities = result[0]['emotion']

                        # Remove 'surprise' and add its probability to 'neutral'
                        if 'surprise' in emotion_probabilities:
                            emotion_probabilities['neutral'] += emotion_probabilities['surprise']
                            del emotion_probabilities['surprise']

                        # Prepare the probabilities in the order of emotion_labels
                        probabilities = [round(float(emotion_probabilities[label]), 2) for label in emotion_labels]

                        # Get the dominant emotion
                        emotion = max(emotion_probabilities, key=emotion_probabilities.get)

                        # Append the result in the desired format
                        results.append(f"{emotion.capitalize()} {probabilities}")
                    except Exception as e:
                        print(f"Error in emotion detection: {e}")

        frame_id += 1

    # Release the video capture object
    cap.release()

    # Evaluate the overall mood
    mood_evaluation = evaluate_overall_mood(results)

    # Return the results and overall mood evaluation
    return {
        "frame_results": results,
        "mood_evaluation": mood_evaluation
    }

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("API:app", host="0.0.0.0", port=8080)