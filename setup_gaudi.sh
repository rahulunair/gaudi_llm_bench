#!/bin/bash

set -e  # Exit on error
set -u  # Exit on undefined variable
set -o pipefail  # Exit on pipe failure

# Configuration
CONTAINER_NAME="inference_perf"
DOCKER_IMAGE="vault.habana.ai/gaudi-docker/1.18.0/ubuntu22.04/habanalabs/pytorch-installer-2.4.0:latest"
MOUNT_PATH="/tmp:/mnt"
HOME_MOUNT="$HOME:/root"
WORKSPACE_DIR=$(pwd)
MIN_DISK_SPACE_GB=100

# Function to check network connectivity
check_network() {
    if ! ping -c 1 8.8.8.8 >/dev/null 2>&1; then
        echo "Error: No internet connectivity"
        return 1
    fi
    if ! docker pull $DOCKER_IMAGE > /dev/null 2>&1; then
        echo "Error: Cannot access Docker registry at vault.habana.ai"
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
        echo "Container '$CONTAINER_NAME' already exists. Removing it..."
        docker rm -f $CONTAINER_NAME
    fi
    return 0
}

# Run pre-flight checks
echo "Running pre-flight checks..."

# Check for HuggingFace token
if [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
    echo "Error: HUGGING_FACE_HUB_TOKEN environment variable is not set"
    exit 1
fi

# Run checks
echo "Checking disk space..."
check_disk_space || exit 1

echo "Checking network connectivity..."
check_network || exit 1

echo "Checking container state..."
check_container_state || exit 1

# Launch container and run setup
echo "Launching container and running setup..."
docker run -d --runtime=habana \
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
    /bin/bash -c "cd /workspace && \
    echo 'Installing optimum-habana...' && \
    pip install 'transformers==4.34.1' && \
    pip install --upgrade-strategy eager 'optimum[habana]==1.14.1' && \
    echo 'Setting up optimum-habana repository...' && \
    if [ -d 'optimum-habana' ]; then \
        (cd optimum-habana && git fetch && git checkout v1.14.0) || exit 1; \
    else \
        git clone https://github.com/huggingface/optimum-habana.git && \
        (cd optimum-habana && git checkout v1.14.0) || exit 1; \
    fi && \
    echo 'Installing additional requirements...' && \
    pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.18.0 && \
    pip install -r optimum-habana/examples/text-generation/requirements.txt && \
    pip install -r optimum-habana/examples/text-generation/requirements_lm_eval.txt && \
    echo 'Setting up HuggingFace cache...' && \
    export HF_HOME=/mnt/huggingface && \
    mkdir -p \$HF_HOME && \
    echo 'Verifying installations...' && \
    python -c 'import torch; print(\"PyTorch version:\", torch.__version__)' && \
    python -c 'import optimum; print(\"Optimum version:\", optimum.__version__)' && \
    python -c 'import deepspeed; print(\"DeepSpeed version:\", deepspeed.__version__)' && \
    echo 'Setup completed successfully!' && \
    tail -f /dev/null"

sleep 5

echo "Waiting for setup to complete..."
timeout=300  # 5 minutes timeout
start_time=$(date +%s)
while true; do
    if docker logs $CONTAINER_NAME 2>&1 | grep -q "Setup completed successfully!"; then
        break
    fi
    if docker logs $CONTAINER_NAME 2>&1 | grep -q "Error\|error\|Failed\|failed"; then
        echo "Error detected in setup:"
        docker logs $CONTAINER_NAME
        docker rm -f $CONTAINER_NAME 2>/dev/null || true
        exit 1
    fi
    current_time=$(date +%s)
    if [ $((current_time - start_time)) -gt $timeout ]; then
        echo "Setup timed out after ${timeout} seconds"
        docker logs $CONTAINER_NAME
        docker rm -f $CONTAINER_NAME 2>/dev/null || true
        exit 1
    fi
    sleep 5
done

# Verify container is running
if ! docker ps -q -f name=$CONTAINER_NAME | grep -q .; then
    echo "Error: Container failed to start or setup failed"
    docker logs $CONTAINER_NAME
    docker rm -f $CONTAINER_NAME 2>/dev/null || true
    exit 1
fi

echo -e "\n✅ Container setup complete!"
echo -e "\nTo run the benchmark:"
echo "1. Run:  ./run_benchmarks.sh"
echo -e "\nTo stop the container when done:"
echo "docker stop $CONTAINER_NAME"