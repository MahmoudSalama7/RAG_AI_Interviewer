# Use an official Python base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install ffmpeg and clean up apt cache in one layer
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies, then clean pip cache
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && rm -rf /root/.cache/pip

# Copy only necessary files and folders
COPY API.py ./
COPY source ./source
COPY weights ./weights
COPY cache ./cache
COPY videos ./videos

# Expose the port FastAPI will run on
EXPOSE 8009

# Start the FastAPI app
CMD ["uvicorn", "API:app", "--host", "0.0.0.0", "--port", "8009"] 