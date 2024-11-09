# Use the official Python image from Docker Hub
FROM nvidia/cuda:12.6.2-runtime-ubuntu24.04





# Install system dependencies
RUN apt update && apt -y upgrade
RUN apt install -y ffmpeg tesseract-ocr libsm6 libxext6 \
    python3 python3-venv python3-ipython python3-pip


    

# Set the working directory in the container
WORKDIR /video_parser

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install Python dependencies
RUN pip install --break-system-packages --no-cache-dir -r requirements.txt



