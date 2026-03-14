# Use an official PyTorch image with CUDA 12.1 support
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

# Set the working directory in the container
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    nano \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
# We upgrade pip first, then install requirements
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create directories for outputs and data so they can be easily mounted as volumes
RUN mkdir -p /workspace/data /workspace/outputs /workspace/configs

# Copy the codebase into the container
COPY . /workspace/

# Set environment variables for Weights & Biases (optional)
# ENV WANDB_API_KEY=your_key_here
# ENV WANDB_PROJECT=CS431-DoRA-vs-PiSSA-LegalSLM

# Default command to keep the container running if started without arguments
CMD ["/bin/bash"]
