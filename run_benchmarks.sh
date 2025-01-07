#!/bin/bash

set -e  # Exit on error
set -u  # Exit on undefined variable
set -o pipefail  # Exit on pipe failure

# Configuration
CONTAINER_NAME="inference_perf"
DOCKER_IMAGE="vault.habana.ai/gaudi-docker/1.18.0/ubuntu22.04/habanalabs/pytorch-installer-2.4.0:latest"
MOUNT_PATH="/scratch-1:/mnt"
HOME_MOUNT="/home/sdp:/root"
WORKSPACE_DIR=$(pwd)
MIN_DISK_SPACE_GB=100  # Minimum required disk space in GB

# Function to check network connectivity
check_network() {
    # Check internet connectivity
    if ! ping -c 1 8.8.8.8 >/dev/null 2>&1; then
        echo "Error: No internet connectivity"
        return 1
    fi

    # Test Docker registry access
    if ! docker pull $DOCKER_IMAGE > /dev/null 2>&1; then
        echo "Error: Cannot access Docker registry at vault.habana.ai"
        echo "Please check your Docker configuration and network connection"
        return 1
    fi

    return 0
}

# Function to check disk space
check_disk_space() {
    local available_space=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ "$available_space" -lt "$MIN_DISK_SPACE_GB" ]; then
        echo "Error: Insufficient disk space. Required: ${MIN_DISK_SPACE_GB}GB, Available: ${available_space}GB"
        return 1
    fi
    return 0
}

# Function to check container state
check_container_state() {
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "Error: Container '$CONTAINER_NAME' already exists"
        echo "Please remove it first: docker rm -f $CONTAINER_NAME"
        return 1
    fi
    return 0
}

# Run pre-flight checks
echo "Running pre-flight checks..."

# Check for HuggingFace token
if [ -z "${HUGGING_FACE_HUB_TOKEN}" ]; then
    echo "Error: HUGGING_FACE_HUB_TOKEN environment variable is not set"
    exit 1
fi

# Check disk space
echo "Checking disk space..."
if ! check_disk_space; then
    exit 1
fi

# Check network connectivity
echo "Checking network connectivity..."
if ! check_network; then
    exit 1
fi

# Check for existing container
echo "Checking for existing containers..."
if ! check_container_state; then
    exit 1
fi

# Launch container for benchmark
echo "Launching container..."
if ! docker run -d --runtime=habana \
    -e HABANA_VISIBLE_DEVICES=all \
    -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
    -e HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN} \
    --cap-add=sys_nice \
    --net=host \
    --ipc=host \
    --name=$CONTAINER_NAME \
    --volume $MOUNT_PATH \
    -v $HOME_MOUNT \
    -v $WORKSPACE_DIR:/workspace \
    $DOCKER_IMAGE \
    /bin/bash -c "
    cd /workspace && \
    python -m venv /workspace/venv && \
    . /workspace/venv/bin/activate && \
    pip install --upgrade-strategy eager optimum[habana] transformers && \
    if [ -d 'optimum-habana' ]; then
        cd optimum-habana && git fetch && git checkout v1.14.0 && cd ..
    else
        git clone https://github.com/huggingface/optimum-habana && \
        cd optimum-habana && git checkout v1.14.0 && cd ..
    fi && \
    pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.18.0 && \
    pip install -r /workspace/optimum-habana/examples/text-generation/requirements.txt && \
    pip install -r /workspace/optimum-habana/examples/text-generation/requirements_lm_eval.txt && \
    export HF_HOME=/mnt/huggingface && \
    python /workspace/run_benchmarks.py 2>&1 | tee /workspace/benchmark.log"; then
    
    echo "Error: Failed to start container"
    exit 1
fi

echo "Container started successfully. Following logs..."
docker logs -f $CONTAINER_NAME
