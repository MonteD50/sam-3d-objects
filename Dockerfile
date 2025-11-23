# Use NVIDIA CUDA base image (matches cu121)
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set environment variables to non-interactive (prevents prompts)
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

# 1. Install System Dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Miniconda (to support Mamba/Conda envs)
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 

# 3. Install Mamba
RUN conda install -c conda-forge mamba -y

# 4. Set up the Application Directory
WORKDIR /app

# 5. Copy Environment File & Create Environment
# We copy just the yaml first to leverage Docker caching
COPY environments/default.yml /app/environments/default.yml

# Create env and ensure shell uses it
RUN mamba env create -f environments/default.yml
# Make the shell activate this environment for future RUN commands
SHELL ["conda", "run", "-n", "sam3d-objects", "/bin/bash", "-c"]

# 6. Install PyTorch & CUDA Dependencies (Your specific exports)
ENV PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
ENV PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"

# 7. Copy the rest of the code
COPY . /app

# Add rembg for background removal
RUN pip install rembg[gpu]
# Pre-download the rembg model so it's cached in the image
RUN python -c "from rembg.bg import download_models; download_models()"
# 8. Run your specific Install & Patch Steps
# Note: We removed '-e' (editable) because in Docker the code is static
RUN pip install '.[dev]'
RUN pip install '.[p3d]'
RUN pip install '.[inference]'
RUN pip install fastapi uvicorn python-multipart

# Run the patching script
RUN chmod +x ./patching/hydra && ./patching/hydra .

# 9. Expose Port and Run
EXPOSE 8000

# Use conda run to ensure the environment is active when container starts
CMD ["conda", "run", "--no-capture-output", "-n", "sam3d-objects", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]