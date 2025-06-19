from fastapi import FastAPI, File, UploadFile
import shutil
import os
from io import BytesIO
from faster_whisper import WhisperModel
import requests

app = FastAPI()

UPLOAD_DIR = "/home/talal/Programming/GP_Files/final/STT/uploads"
LOG_FILE = "transcriptions.log"
os.makedirs(UPLOAD_DIR, exist_ok=True)
# os.makedirs("models", exist_ok=True)  

# Load Whisper model at startup
print("Loading Whisper model...")
local_model_path = "/home/talal/Programming/GP_Files/final/STT/models/models--Systran--faster-whisper-small/snapshots/536b0662742c02347bc0e980a01041f333bce120"
print(f"Loading model from local path: {local_model_path}")
model = WhisperModel(local_model_path, device="cuda", compute_type="float16")
print("Model loaded successfully.")

# LLM_API_URL = os.getenv("LLM_API_URL", "http://localhost:8001/interview")

@app.post("/upload_and_transcribe/")
async def upload_and_transcribe(file: UploadFile = File(...)):
    audio_data = await file.read()
    audio_buffer = BytesIO(audio_data)
    # with open("debug_uploaded.wav", "wb") as f:
    #     f.write(audio_data)
    # print("[API DEBUG] Saved uploaded audio as debug_uploaded.wav")

    try:
        segments, _ = model.transcribe(audio_buffer, vad_filter=False)
        transcription = " ".join([segment.text for segment in segments])
        result = {"transcription": transcription}
        # print(f"[API DEBUG] Transcription result: {result}")
        
        # if LLM_API_URL:
        #     try:
        #         llm_resp = requests.post(LLM_API_URL, json={"transcription": transcription})
        #         if llm_resp.ok:
        #             result["llm_response"] = llm_resp.json().get("llm_response")
        #             result["avatar_video_url"] = llm_resp.json().get("avatar_video_url")
        #         else:
        #             result["llm_error"] = llm_resp.text
        #     except Exception as e:
        #         result["llm_error"] = str(e)
        return result
    except Exception as e:
        return {"error": str(e)}