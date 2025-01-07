# Gaudi2 LLM Benchmarking Suite

A  benchmarking suite for running LLM inference on Habana Gaudi2 cards. This suite supports both quantized (FP8) and BF16 runs, and handles multi-card configurations automatically.



## Configuration

Models and their configurations are defined in `benchmark_config.json`:

```json
{
    "models": [
        {
            "name": "model-name/path",
            "size": 11,  // Size in billions of parameters
            "cards_required": 1,  // Number of cards needed
            "configs": [
                {
                    "input_tokens": 128,
                    "output_tokens": 128,
                    "batch_size": 128
                }
                // ... more configurations
            ]
        }
        // ... more models
    ]
}
```

## Usage

1. Set your HuggingFace token as an environment variable:
```bash
export HUGGING_FACE_HUB_TOKEN='your-token-here'
```

2. Make the script executable and run:
```bash
chmod +x run_benchmarks.sh
./run_benchmarks.sh
```

3. Monitor progress:
```bash
# Attach to tmux session
tmux attach-session -t benchmark_session

# Detach without stopping: Press Ctrl+B, then D
```

## Output

Results are saved to `benchmark_results.csv` with the following columns:
- timestamp
- model
- input_tokens
- output_tokens
- batch_size
- throughput
- time
- mode (quantized/bf16)

## Docker Environment

The benchmarks run in a Habana PyTorch container with:
- Ubuntu 22.04
- PyTorch 2.4.0
- Optimum Habana
- DeepSpeed

## File Structure

- `run_benchmarks.sh`: Main shell script for container setup and execution
- `run_benchmarks.py`: Python script handling the benchmark logic
- `benchmark_config.json`: Configuration file for models and parameters
- `benchmark_results.csv`: Output file with benchmark results
