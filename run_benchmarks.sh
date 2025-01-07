#!/bin/bash

set -e  # Exit on error
set -u  # Exit on undefined variable
set -o pipefail  # Exit on pipe failure

# Configuration
CONTAINER_NAME="inference_perf"
BASE_DOCKER_IMAGE="vault.habana.ai/gaudi-docker/1.18.0/ubuntu22.04/habanalabs/pytorch-installer-2.4.0:latest"
CACHED_IMAGE_NAME="gaudi-bench:cached"
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
    # Only check essential services that don't require authentication
    local urls=("huggingface.co" "github.com")
    for url in "${urls[@]}"; do
        if ! curl --silent --head --fail "https://${url}" >/dev/null; then
            echo "Error: Cannot connect to ${url}"
            echo "Please check your network connection and try again"
            return 1
        fi
    done

    # Test Docker registry access directly
    if ! docker pull $BASE_DOCKER_IMAGE > /dev/null 2>&1; then
        echo "Error: Cannot access Docker registry at vault.habana.ai"
        echo "Please check your Docker configuration and network connection"
        return 1
    fi

    return 0
}

# Function to check container state
check_container_state() {
    local container_id=$(docker ps -aqf "name=$CONTAINER_NAME")
    if [ -z "$container_id" ]; then
        echo "Container does not exist"
        return 0
    fi

    local state=$(docker inspect -f '{{.State.Status}}' "$container_id")
    local running=$(docker inspect -f '{{.State.Running}}' "$container_id")
    local exit_code=$(docker inspect -f '{{.State.ExitCode}}' "$container_id")
    
    echo "Container state: $state"
    echo "Running: $running"
    echo "Exit code: $exit_code"
    
    if [ "$state" = "created" ]; then
        echo "Found a stale container that was never started"
        docker rm "$container_id"
        return 0
    elif [ "$state" = "exited" ]; then
        if [ "$exit_code" = "0" ]; then
            echo "Found a successfully completed container"
            docker rm "$container_id"
            return 0
        else
            echo "Found a failed container (exit code: $exit_code)"
            echo "Preserving container for debugging"
            echo "To remove it manually: docker rm $container_id"
            return 1
        fi
    elif [ "$running" = "true" ]; then
        echo "Error: Container is already running"
        return 1
    fi
}

# Function to build and cache container
build_cached_container() {
    echo "Building cached container with dependencies..."
    
    # Start temporary container
    local temp_container="temp_build_$$"
    if ! docker run -d --name=$temp_container \
        --runtime=habana \
        -e HABANA_VISIBLE_DEVICES=all \
        -v $WORKSPACE_DIR:/workspace \
        $BASE_DOCKER_IMAGE \
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
        sleep 5"; then
        echo "Error: Failed to start build container"
        return 1
    fi

    # Wait for installation to complete
    echo "Waiting for package installation to complete..."
    sleep 10
    while docker ps -q --filter "name=$temp_container" >/dev/null; do
        sleep 5
    done

    # Check if build was successful
    if [ "$(docker inspect -f '{{.State.ExitCode}}' $temp_container)" != "0" ]; then
        echo "Error: Build container failed"
        docker logs $temp_container
        docker rm $temp_container
        return 1
    fi

    # Commit the container as a new image
    echo "Committing container as new image: $CACHED_IMAGE_NAME"
    docker commit $temp_container $CACHED_IMAGE_NAME
    docker rm $temp_container
    
    echo "Successfully created cached image"
    return 0
}

# Function to check if cached image exists and is valid
check_cached_image() {
    if ! docker image inspect $CACHED_IMAGE_NAME >/dev/null 2>&1; then
        echo "Cached image not found"
        return 1
    fi
    
    # Add additional validation if needed
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

# Check network connectivity (including Docker registry)
echo "Checking network connectivity..."
if ! check_network; then
    exit 1
fi

# Check for existing container
echo "Checking for existing containers..."
if ! check_container_state; then
    echo "Please check the existing container before proceeding"
    exit 1
fi

# Check for cached image or build it
if ! check_cached_image; then
    echo "Building new cached image..."
    if ! build_cached_container; then
        echo "Failed to build cached image, falling back to base image"
        DOCKER_IMAGE=$BASE_DOCKER_IMAGE
    else
        DOCKER_IMAGE=$CACHED_IMAGE_NAME
    fi
else
    echo "Using cached image"
    DOCKER_IMAGE=$CACHED_IMAGE_NAME
fi

# Launch container for benchmark
echo "Launching container for benchmark..."
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
    . /workspace/venv/bin/activate && \
    export HF_HOME=/mnt/huggingface && \
    python /workspace/run_benchmarks.py 2>&1 | tee /workspace/benchmark.log"; then
    
    echo "Error: Failed to start container"
    exit 1
fi

echo -e "\nContainer started successfully!"
echo -e "\nShowing live container output (Ctrl+C to stop watching, container will continue running):"
echo "================================================================================"

# Show initial output
sleep 2  # Give container a moment to start
docker logs -f $CONTAINER_NAME &
LOGS_PID=$!

# Function to handle Ctrl+C
ctrl_c() {
    kill $LOGS_PID 2>/dev/null
    echo -e "\n\nStopped watching logs. Container is still running."
    echo "To view logs again:"
    echo "1. Container logs:    docker logs -f $CONTAINER_NAME"
    echo "2. Benchmark log:     tail -f benchmark.log"
    echo "3. Container status:  docker ps | grep $CONTAINER_NAME"
    echo ""
    echo "Monitoring container status..."
}

# Set up Ctrl+C handler
trap ctrl_c INT

# Monitor container status while showing logs
while docker ps | grep -q $CONTAINER_NAME; do
    wait $LOGS_PID 2>/dev/null || true
    
    # If logs process died but container is still running, restart it
    if ! kill -0 $LOGS_PID 2>/dev/null; then
        docker logs -f $CONTAINER_NAME &
        LOGS_PID=$!
    fi
    
    sleep 5
done

# Container has stopped - check status
kill $LOGS_PID 2>/dev/null || true
echo -e "\nContainer has finished."
EXIT_CODE=$(docker inspect $CONTAINER_NAME --format='{{.State.ExitCode}}')

# Check both container exit code and benchmark log for errors
BENCHMARK_FAILED=0
if grep -q "CRITICAL" benchmark.log 2>/dev/null; then
    BENCHMARK_FAILED=1
fi

if [ "$EXIT_CODE" = "0" ] && [ "$BENCHMARK_FAILED" = "0" ]; then
    echo "Benchmark completed successfully!"
    echo "Results are in benchmark_results.csv"
    echo "Full logs are in benchmark.log"
    docker rm $CONTAINER_NAME
else
    echo "Benchmark failed!"
    if [ "$BENCHMARK_FAILED" = "1" ]; then
        echo "Critical errors found in benchmark.log:"
        grep "CRITICAL" benchmark.log
    fi
    echo "Container exited with code $EXIT_CODE"
    echo "Last 50 lines of container logs:"
    docker logs --tail 50 $CONTAINER_NAME
    echo ""
    echo "Container preserved for debugging"
    echo "To remove it manually: docker rm $CONTAINER_NAME"
    exit 1
fi
