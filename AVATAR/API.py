from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import FileResponse
import shutil
import os
import sys
import time
import subprocess
from lipsync import LipSync
import contextlib
from fastapi.staticfiles import StaticFiles
import uuid


os.environ['FFMPEG_LOGLEVEL'] = 'quiet'

app = FastAPI()

# Define base directory using the current file's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define paths relative to BASE_DIR
CACHE_DIR = os.path.join(BASE_DIR, "cache")
STATIC_VIDEO = os.path.join(BASE_DIR, "source", "480.mp4")  # Pre-cached video
OUTPUT_VIDEO = "result.mp4"
UPLOAD_AUDIO_DIR = os.path.join(BASE_DIR, "source")
UPLOAD_AUDIO_PATH = os.path.join(UPLOAD_AUDIO_DIR, "input_audio.wav")  # Save uploaded audio here
VIDEOS_DIR = os.path.join(BASE_DIR, "videos")
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")

os.makedirs(os.path.dirname(UPLOAD_AUDIO_DIR), exist_ok=True)
# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(VIDEOS_DIR, exist_ok=True)
app.mount("/videos", StaticFiles(directory=VIDEOS_DIR), name="videos")

# Initialize LipSync with caching enabled
lip = LipSync(
    model='wav2lip',
    checkpoint_path=os.path.join(WEIGHTS_DIR, "wav2lip.pth"),
    nosmooth=True,
    device='cuda',
    cache_dir=CACHE_DIR,
    img_size=96,
    save_cache=True,
    ffmpeg_loglevel='quiet', 
    fps=25,
)

@app.post("/sync")
async def sync_audio(request: Request, audio: UploadFile = File(...)):
    # Save the uploaded audio file
    with open(UPLOAD_AUDIO_PATH, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)
    
    # Suppress all output (including subprocess output)
    with open(os.devnull, 'w') as devnull:
        # Redirect stdout and stderr to devnull
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        
        # Redirect file descriptors 1 (stdout) and 2 (stderr) to devnull
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            # Measure execution time
            start_time = time.time()
            
            # Generate a unique filename for the output video
            unique_id = str(uuid.uuid4())
            output_filename = f"{unique_id}.mp4"
            output_path = os.path.join(VIDEOS_DIR, output_filename)
            
            # Run the lip-sync process with suppressed output
            lip.sync(STATIC_VIDEO, UPLOAD_AUDIO_PATH, output_path)
            
            end_time = time.time()
        
        # Restore stdout and stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    
    # Print execution time
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    
    # Return the video URL as JSON (full URL)
    base_url = str(request.base_url).rstrip("/")
    video_url = f"{base_url}/videos/{output_filename}"
    return {"video_url": video_url}