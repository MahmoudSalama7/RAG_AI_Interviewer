from lipsync import LipSync
import torch
import time

# Check if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

try:
    start_time = time.time()
    lip = LipSync(
        model='wav2lip',
        checkpoint_path='/home/talal/Programming/GP_Files/AVATAR/weights/wav2lip.pth',
        nosmooth=True,
        device=device,
        cache_dir='/home/talal/Programming/GP_Files/AVATAR/cache',
        img_size=96,  
        save_cache=True,
        fps=25,  
    )

    lip.sync(
        '/home/talal/Programming/GP_Files/AVATAR/source/AVATAR.mp4',
        '/home/talal/Programming/GP_Files/AVATAR/source/audio1.mp3',
        '/home/talal/Programming/GP_Files/AVATAR/result8.mp4',
    )
    end_time = time.time()
    print(f"Processing time: {end_time - start_time:.2f} seconds")
except Exception as e:
    print(f"An error occurred: {e}")
