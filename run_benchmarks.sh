#!/bin/bash

# Configuration
CONTAINER_NAME="inference_perf"
DOCKER_IMAGE="vault.habana.ai/gaudi-docker/1.18.0/ubuntu22.04/habanalabs/pytorch-installer-2.4.0:latest"
MOUNT_PATH="/scratch-1:/mnt"
HOME_MOUNT="/home/sdp:/root"
TMUX_SESSION="benchmark_session"

# Check for HuggingFace token
if [ -z "${HUGGING_FACE_HUB_TOKEN}" ]; then
    echo "Error: HUGGING_FACE_HUB_TOKEN environment variable is not set"
    echo "Please set it first:"
    echo "export HUGGING_FACE_HUB_TOKEN='your-token-here'"
    exit 1
fi

# Create a new tmux session or attach to existing one
tmux new-session -d -s $TMUX_SESSION 2>/dev/null || true

# Stop and remove existing container if it exists
tmux send-keys -t $TMUX_SESSION "docker stop $CONTAINER_NAME 2>/dev/null" C-m
tmux send-keys -t $TMUX_SESSION "docker rm $CONTAINER_NAME 2>/dev/null" C-m

# Launch the container with all setup commands
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
    cd /workspace && \
    pip install --upgrade-strategy eager optimum[habana] && \
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
