import csv
import json
import os
import subprocess
import time
from datetime import datetime
from typing import Dict, Optional


def run_command(cmd):
    process = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()
    return stdout.decode(), stderr.decode(), process.returncode


def run_benchmark_bf16(
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    batch_size: int,
    cards_required: int,
) -> Optional[Dict]:
    """Run benchmark in BF16 mode without quantization"""
    print(f"\nRunning BF16 benchmark for {model_name}")
    print(
        f"Input tokens: {input_tokens}, Output tokens: {output_tokens}, Batch size: {batch_size}"
    )

    base_cmd = (
        f"python ../gaudi_spawn.py --use_deepspeed --world_size {cards_required}"
        if cards_required > 1
        else "python"
    )
    cmd = f"""HF_DATASETS_TRUST_REMOTE_CODE=true QUANT_CONFIG=./quantization_config/maxabs_quant.json TQDM_DISABLE=1 {base_cmd} \
    run_generation.py --model_name_or_path {model_name} \
    --attn_softmax_bf16 \
    --trim_logits \
    --warmup 2 \
    --use_kv_cache \
    --use_hpu_graphs \
    --limit_hpu_graphs \
    --bucket_size=128 \
    --bucket_internal \
    --bf16 \
    --max_input_tokens {input_tokens} \
    --max_new_tokens {output_tokens} \
    --batch_size {batch_size} \
    --flash_attention_causal_mask \
    --use_flash_attention \
    --flash_attention_recompute"""

    print(f"Running command: {cmd}")
    start_time = time.time()
    stdout, stderr, returncode = run_command(cmd)
    end_time = time.time()

    if returncode != 0:
        print(f"Error during BF16 benchmark: {stderr}")
        print(f"Command output: {stdout}")
        return None

    throughput = None
    for line in stdout.split("\n"):
        if "tokens/sec" in line:
            try:
                throughput = float(line.split(":")[1].strip())
            except Exception as e:
                print(f"Error parsing throughput: {e}")
                print(f"Line content: {line}")
                pass

    return {
        "model": model_name,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "batch_size": batch_size,
        "throughput": throughput,
        "time": end_time - start_time,
        "mode": "bf16",
    }


def quantize_model(model_name: str, cards_required: int) -> bool:
    print(f"\nQuantizing model: {model_name}")
    base_cmd = (
        f"python ../gaudi_spawn.py --use_deepspeed --world_size {cards_required}"
        if cards_required > 1
        else "python"
    )

    cmd = f"""HF_DATASETS_TRUST_REMOTE_CODE=true QUANT_CONFIG=./quantization_config/maxabs_measure.json \
    TQDM_DISABLE=1 {base_cmd} \
    run_lm_eval.py --model_name_or_path {model_name} \
    --attn_softmax_bf16 \
    --trim_logits \
    --warmup 2 \
    --use_kv_cache \
    --use_hpu_graphs \
    --limit_hpu_graphs \
    --bucket_size=128 \
    --bucket_internal \
    --bf16 \
    --batch_size 1 \
    --flash_attention_causal_mask \
    --use_flash_attention \
    --flash_attention_recompute \
    -o quant_measure_{model_name.split('/')[-1]}.txt 2>&1 | tee -a /home/log_measur_quant.txt"""

    print(f"Running quantization command: {cmd}")
    stdout, stderr, returncode = run_command(cmd)
    if returncode != 0:
        print(f"Error during quantization: {stderr}")
        print(f"Command output: {stdout}")
        return False
    return True


def run_benchmark_quantized(
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    batch_size: int,
    cards_required: int,
) -> Optional[Dict]:
    print(f"\nRunning quantized benchmark for {model_name}")
    print(
        f"Input tokens: {input_tokens}, Output tokens: {output_tokens}, Batch size: {batch_size}"
    )

    base_cmd = (
        f"python ../gaudi_spawn.py --use_deepspeed --world_size {cards_required}"
        if cards_required > 1
        else "python"
    )
    cmd = f"""HF_DATASETS_TRUST_REMOTE_CODE=true QUANT_CONFIG=./quantization_config/maxabs_quant.json \
    TQDM_DISABLE=1 {base_cmd} run_generation.py --model_name_or_path {model_name} \
    --attn_softmax_bf16 \
    --trim_logits \
    --warmup 2 \
    --use_kv_cache \
    --use_hpu_graphs \
    --limit_hpu_graphs \
    --bucket_size=128 \
    --bucket_internal \
    --bf16 \
    --max_input_tokens {input_tokens} \
    --max_new_tokens {output_tokens} \
    --batch_size {batch_size} \
    --flash_attention_causal_mask \
    --use_flash_attention \
    --flash_attention_recompute"""

    print(f"Running quantized benchmark command: {cmd}")
    start_time = time.time()
    stdout, stderr, returncode = run_command(cmd)
    end_time = time.time()

    if returncode != 0:
        print(f"Error during quantized benchmark: {stderr}")
        print(f"Command output: {stdout}")
        return None

    throughput = None
    for line in stdout.split("\n"):
        if "tokens/sec" in line:
            try:
                throughput = float(line.split(":")[1].strip())
            except Exception as e:
                print(f"Error parsing throughput: {e}")
                print(f"Line content: {line}")
                pass

    return {
        "model": model_name,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "batch_size": batch_size,
        "throughput": throughput,
        "time": end_time - start_time,
        "mode": "quantized",
    }


def main():
    # Check for HuggingFace token
    if not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        print("Error: HUGGING_FACE_HUB_TOKEN environment variable is not set")
        print("Please set it in your environment before running the script:")
        print("export HUGGING_FACE_HUB_TOKEN='your-token-here'")
        return

    # Load configuration
    config_path = "/workspace/benchmark_config.json"
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at {config_path}")
        return

    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return

    # Sort models by size
    models = sorted(config["models"], key=lambda x: x["size"])

    # Prepare results file
    results_file = "/workspace/benchmark_results.csv"
    fieldnames = [
        "timestamp",
        "model",
        "input_tokens",
        "output_tokens",
        "batch_size",
        "throughput",
        "time",
        "mode",
    ]

    try:
        with open(results_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
    except Exception as e:
        print(f"Error creating results file: {e}")
        return

    # Run benchmarks for each model
    for model in models:
        print(f"\n{'='*50}")
        print(f"Starting benchmarks for {model['name']} ({model['size']}B)")
        print(f"{'='*50}")

        # Try quantization first
        quant_success = quantize_model(model["name"], model["cards_required"])
        
        if not quant_success:
            print(f"\nQuantization failed for {model['name']}, will run in BF16 mode only")

        # For each configuration, run either quantized or BF16 benchmark
        for cfg in model["configs"]:
            result = None
            
            # Try quantized run if quantization was successful
            if quant_success:
                print("\nAttempting quantized run...")
                result = run_benchmark_quantized(
                    model["name"],
                    cfg["input_tokens"],
                    cfg["output_tokens"],
                    cfg["batch_size"],
                    model["cards_required"],
                )
                
                if result:
                    print("Quantized run successful!")
                else:
                    print("Quantized run failed, falling back to BF16...")

            # If either quantization failed or quantized run failed, try BF16
            if not quant_success or not result:
                print("\nRunning BF16 version...")
                result = run_benchmark_bf16(
                    model["name"],
                    cfg["input_tokens"],
                    cfg["output_tokens"],
                    cfg["batch_size"],
                    model["cards_required"],
                )

            if result:
                result["timestamp"] = datetime.now().isoformat()
                try:
                    with open(results_file, "a", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writerow(result)
                except Exception as e:
                    print(f"Error writing results: {e}")

                print(f"\nResults for {model['name']}:")
                print(f"Mode: {result['mode']}")
                print(f"Input tokens: {cfg['input_tokens']}")
                print(f"Output tokens: {cfg['output_tokens']}")
                print(f"Batch size: {cfg['batch_size']}")
                print(f"Throughput: {result['throughput']}")
                print("-" * 50)
            else:
                print(f"Failed to get results for {model['name']} with config: {cfg}")
                print("Both quantized and BF16 runs failed for this configuration")


if __name__ == "__main__":
    main()
