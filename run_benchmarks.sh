#!/bin/bash

set -e

CONTAINER_NAME="inference_perf"

# Check if container exists and is running
if ! docker ps -q -f name=$CONTAINER_NAME | grep -q .; then
    echo "Error: Container $CONTAINER_NAME is not running"
    echo "Please run setup_gaudi.sh first"
    exit 1
fi

# Execute benchmark
echo "Running benchmark..."
docker exec -it $CONTAINER_NAME /bin/bash -c "cd /workspace && python run_benchmarks.py"

echo "Benchmark complete!"