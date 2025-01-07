#!/bin/bash

set -e  # Exit on error

# Configuration
CONTAINER_NAME="inference_perf"
DOCKER_IMAGE="vault.habana.ai/gaudi-docker/1.18.0/ubuntu22.04/habanalabs/pytorch-installer-2.4.0:latest"
MOUNT_PATH="/scratch-1:/mnt"
HOME_MOUNT="/home/sdp:/root"
TMUX_SESSION="benchmark_session"

# Cleanup function
cleanup() {
    echo "Cleaning up..."
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true
    tmux kill-session -t $TMUX_SESSION 2>/dev/null || true
}

# Set trap for cleanup
trap cleanup EXIT INT TERM

# Check for HuggingFace token
if [ -z "${HUGGING_FACE_HUB_TOKEN}" ]; then
    echo "Error: HUGGING_FACE_HUB_TOKEN environment variable is not set"
    echo "Please set it first:"
    echo "export HUGGING_FACE_HUB_TOKEN='your-token-here'"
    exit 1
fi

# Check if docker is available
if ! command -v docker &> /dev/null; then
    echo "Error: docker is not installed or not in PATH"
    exit 1
fi

# Check if tmux is available
if ! command -v tmux &> /dev/null; then
    echo "Error: tmux is not installed or not in PATH"
    exit 1
fi

# Check if Habana runtime is available
if ! docker info | grep -i "habana" &> /dev/null; then
    echo "Error: Habana runtime not found in Docker"
    exit 1
fi

# Check if hl-smi is available
if ! command -v hl-smi &> /dev/null; then
    echo "Error: hl-smi not found. Are Gaudi drivers installed?"
    exit 1
fi

# Check Gaudi cards status
echo "Checking Gaudi cards..."
if ! hl-smi &> /dev/null; then
    echo "Error: Failed to query Gaudi cards status"
    exit 1
fi

# Extract card count and driver version
card_count=$(hl-smi | grep "HL-225" | wc -l)
driver_version=$(hl-smi | grep "Driver Version:" | awk '{print $3}')

echo "Found ${card_count} Gaudi cards"
echo "Driver version: ${driver_version}"

# Check if required directories exist
if [ ! -d "/scratch-1" ]; then
    echo "Error: /scratch-1 directory not found"
    exit 1
fi

if [ ! -d "/home/sdp" ]; then
    echo "Error: /home/sdp directory not found"
    exit 1
fi

# Check if benchmark config exists
if [ ! -f "benchmark_config.json" ]; then
    echo "Error: benchmark_config.json not found in current directory"
    exit 1
fi

# Check if run_benchmarks.py exists
if [ ! -f "run_benchmarks.py" ]; then
    echo "Error: run_benchmarks.py not found in current directory"
    exit 1
fi

# Create a new tmux session or attach to existing one
tmux new-session -d -s $TMUX_SESSION 2>/dev/null || true

# Stop and remove existing container if it exists
tmux send-keys -t $TMUX_SESSION "docker stop $CONTAINER_NAME 2>/dev/null || true" C-m
tmux send-keys -t $TMUX_SESSION "docker rm $CONTAINER_NAME 2>/dev/null || true" C-m

# Pull the Docker image first
echo "Pulling Docker image..."
if ! docker pull $DOCKER_IMAGE; then
    echo "Error: Failed to pull Docker image"
    exit 1
fi

# Launch the container with all setup commands
echo "Launching container..."
tmux send-keys -t $TMUX_SESSION "docker run -it --runtime=habana \
    -e HABANA_VISIBLE_DEVICES=all \
    -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
    -e HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN} \
    --cap-add=sys_nice \
    --net=host \
    --ipc=host \
    --name=$CONTAINER_NAME \
    --volume $MOUNT_PATH \
    -v $HOME_MOUNT \
    -v $(pwd):/workspace \
    $DOCKER_IMAGE \
    /bin/bash -c \"
    set -e && \
    cd /workspace && \
    pip install --upgrade-strategy eager optimum[habana] transformers && \
    git clone https://github.com/huggingface/optimum-habana && \
    cd optimum-habana && git checkout v1.14.0 && \
    pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.18.0 && \
    cd examples/text-generation/ && \
    pip install -r requirements.txt && \
    pip install -r requirements_lm_eval.txt && \
    export HF_HOME=/mnt/huggingface && \
    python /workspace/run_benchmarks.py\"" C-m

echo "Benchmark session started in tmux. To attach to the session:"
echo "tmux attach-session -t $TMUX_SESSION"
echo ""
echo "To detach from the session once attached: press Ctrl+B, then D"
echo "To view session output later: tmux attach-session -t $TMUX_SESSION"
echo ""
echo "To check container logs:"
echo "docker logs -f $CONTAINER_NAME"
