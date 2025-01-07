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

echo "Running pre-flight checks..."

# Function to check disk space
check_disk_space() {
    local path=$1
    local available_space=$(df -BG "$path" | awk 'NR==2 {gsub("G","",$4); print $4}')
    if [ "$available_space" -lt "$MIN_DISK_SPACE_GB" ]; then
        echo "Error: Insufficient disk space in $path"
        echo "Available: ${available_space}GB, Required: ${MIN_DISK_SPACE_GB}GB"
        return 1
    fi
    return 0
}

# Function to check network connectivity
check_network() {
    local urls=("huggingface.co" "github.com" "vault.habana.ai")
    for url in "${urls[@]}"; do
        if ! curl --silent --head --fail "https://${url}" >/dev/null; then
            echo "Error: Cannot connect to ${url}"
            echo "Please check your network connection and try again"
            return 1
        fi
    done
    return 0
}

# Check for HuggingFace token
if [ -z "${HUGGING_FACE_HUB_TOKEN}" ]; then
    echo "Error: HUGGING_FACE_HUB_TOKEN environment variable is not set"
    echo "Please set it first:"
    echo "export HUGGING_FACE_HUB_TOKEN='your-token-here'"
    exit 1
fi

# Check if required files exist
if [ ! -f "benchmark_config.json" ]; then
    echo "Error: benchmark_config.json not found in current directory"
    exit 1
fi

if [ ! -f "run_benchmarks.py" ]; then
    echo "Error: run_benchmarks.py not found in current directory"
    exit 1
fi

# Check if docker is available
if ! command -v docker &> /dev/null; then
    echo "Error: docker is not installed or not in PATH"
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

# Check disk space in critical paths
echo "Checking disk space..."
for path in "/scratch-1" "/home/sdp" "$WORKSPACE_DIR"; do
    if ! check_disk_space "$path"; then
        exit 1
    fi
done

# Check network connectivity
echo "Checking network connectivity..."
if ! check_network; then
    exit 1
fi

# Pull the Docker image
echo "Pulling Docker image..."
if ! docker pull $DOCKER_IMAGE; then
    echo "Error: Failed to pull Docker image"
    exit 1
fi

# Stop and remove existing container if it exists
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true

# Launch the container
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
    pip install --upgrade-strategy eager optimum[habana] transformers && \
    git clone https://github.com/huggingface/optimum-habana && \
    cd optimum-habana && git checkout v1.14.0 && \
    pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.18.0 && \
    cd examples/text-generation/ && \
    pip install -r requirements.txt && \
    pip install -r requirements_lm_eval.txt && \
    export HF_HOME=/mnt/huggingface && \
    python /workspace/run_benchmarks.py 2>&1 | tee /workspace/benchmark.log"; then
    
    echo "Error: Failed to start container"
    exit 1
fi

echo -e "\nContainer started. Use these commands to monitor progress:"
echo "1. View live container logs:"
echo "   docker logs -f $CONTAINER_NAME"
echo ""
echo "2. View benchmark log:"
echo "   tail -f benchmark.log"
echo ""
echo "3. Check container status:"
echo "   docker ps | grep $CONTAINER_NAME"

# Monitor container status with timeout
echo -e "\nMonitoring container status..."
TIMEOUT=259200  # 72 hours in seconds
START_TIME=$(date +%s)

while docker ps | grep -q $CONTAINER_NAME; do
    CURRENT_TIME=$(date +%s)
    ELAPSED_TIME=$((CURRENT_TIME - START_TIME))
    
    if [ $ELAPSED_TIME -gt $TIMEOUT ]; then
        echo "Error: Benchmark exceeded maximum runtime of 72 hours"
        docker stop $CONTAINER_NAME
        docker logs --tail 100 $CONTAINER_NAME > benchmark_timeout.log
        echo "Last 100 lines of logs saved to benchmark_timeout.log"
        exit 1
    fi
    
    echo -e "\nContainer is running. Recent output:"
    docker logs --tail 5 $CONTAINER_NAME 2>&1
    sleep 30
done

# Container has stopped - check status
EXIT_CODE=$(docker inspect $CONTAINER_NAME --format='{{.State.ExitCode}}')

if [ "$EXIT_CODE" = "0" ]; then
    echo "Benchmark completed successfully!"
    echo "Results are in benchmark_results.csv"
    echo "Full logs are in benchmark.log"
else
    echo "Container exited with error code $EXIT_CODE"
    echo "Last 50 lines of container logs:"
    docker logs --tail 50 $CONTAINER_NAME
fi

# Cleanup
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true
