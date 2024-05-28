# Use an official Python runtime as a parent image
FROM python:3.10-slim-bullseye

# Set the working directory in the container
WORKDIR /usr/src/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-dev \
    intel-opencl-icd \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    tk \
    && rm -rf /var/lib/apt/lists/*

# Install JupyterLab
RUN pip install --upgrade pip && pip install --no-cache-dir jupyterlab

# Copy only the necessary file
COPY people-counting.ipynb /usr/src/app/

# Make port 8888 available to the world outside this container
EXPOSE 8888

# Run JupyterLab when the container launches
CMD ["jupyter-lab", "people-counting.ipynb", "--ip=0.0.0.0", "--port=8888","--allow-root"]
