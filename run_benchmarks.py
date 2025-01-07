import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, Optional, List
import signal
from transformers import AutoModelForCausalLM, AutoTokenizer


def signal_handler(signum, frame):
    """Handle cleanup on interrupt"""
    print("\n\nInterrupt received. Cleaning up...")
    try:
        subprocess.run("docker stop inference_perf", shell=True)
        subprocess.run("docker rm inference_perf", shell=True)
    except Exception as e:
        print(f"Error during cleanup: {e}")
    sys.exit(1)


def validate_config(config: Dict) -> bool:
    """Validate the configuration file structure"""
    try:
        required_fields = ["models"]
        for field in required_fields:
            if field not in config:
                print(f"Error: Missing required field '{field}' in config")
                return False

        for model in config["models"]:
            model_fields = ["name", "size", "cards_required", "configs"]
            for field in model_fields:
                if field not in model:
                    print(f"Error: Missing required field '{field}' in model config")
                    return False
                
            if not isinstance(model["cards_required"], int) or model["cards_required"] < 1:
                print(f"Error: Invalid cards_required for {model['name']}")
                return False

            for cfg in model["configs"]:
                config_fields = ["input_tokens", "output_tokens", "batch_size"]
                for field in config_fields:
                    if field not in cfg:
                        print(f"Error: Missing required field '{field}' in model config")
                        return False
                    if not isinstance(cfg[field], int) or cfg[field] < 1:
                        print(f"Error: Invalid {field} value in config")
                        return False

        return True
    except Exception as e:
        print(f"Error validating config: {e}")
        return False


def download_models(models: List[Dict]) -> bool:
    """
    Pre-download all models to ensure availability and permissions.
    Returns True if all models are downloaded successfully, False otherwise.
    """
    print("\nPre-downloading models...")
    success = True
    failed_models = []
    
    for model in models:
        model_name = model["name"]
        print(f"\nDownloading {model_name}...")
        try:
            # Try to load tokenizer first as it's smaller
            print(f"Downloading tokenizer for {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_auth_token=os.environ.get("HUGGING_FACE_HUB_TOKEN"),
                timeout=60  # Add timeout
            )
            
            # Then try to load model config (without downloading weights)
            print(f"Validating model config for {model_name}")
            config = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_auth_token=os.environ.get("HUGGING_FACE_HUB_TOKEN"),
                config_only=True,
                timeout=60  # Add timeout
            )
            
            # Validate model size if available in config
            if hasattr(config, "num_parameters"):
                expected_size = model["size"] * 1e9  # Convert B to parameters
                actual_size = config.num_parameters
                if abs(expected_size - actual_size) / expected_size > 0.1:  # 10% tolerance
                    print(f"Warning: Model size mismatch for {model_name}")
                    print(f"Expected: {model['size']}B, Actual: {actual_size/1e9:.1f}B")
            
            print(f"✓ Successfully validated {model_name}")
            
        except Exception as e:
            print(f"✗ Error downloading {model_name}: {str(e)}")
            failed_models.append(model_name)
            success = False
    
    if failed_models:
        print("\nFailed models:")
        for model in failed_models:
            print(f"  - {model}")
    
    return success


def check_gaudi_cards(required_cards: int) -> bool:
    """Check if required number of Gaudi cards are available and healthy"""
    try:
        # Check if hl-smi is available
        result = subprocess.run(["which", "hl-smi"], capture_output=True, text=True)
        if result.returncode != 0:
            print("Error: hl-smi not found. Are Gaudi drivers installed?")
            return False

        # Get hl-smi output
        result = subprocess.run(["hl-smi"], capture_output=True, text=True)
        if result.returncode != 0:
            print("Error running hl-smi command")
            return False

        output = result.stdout

        # Check driver version
        if "Driver Version:" not in output:
            print("Error: Could not find Gaudi driver version")
            return False

        # Count available cards
        card_count = output.count("HL-225")
        if card_count < required_cards:
            print(f"Error: Not enough Gaudi cards. Required: {required_cards}, Found: {card_count}")
            return False

        # Parse card details
        cards = []
        for line in output.split('\n'):
            if 'HL-225' in line:
                cards.append(line)

        # Check each card's status
        for i, card in enumerate(cards):
            # Check temperature (example threshold: 85°C)
            temp = int(card.split()[2].rstrip('C'))
            if temp > 85:
                print(f"Warning: Card {i} temperature is high: {temp}°C")

            # Check power usage
            power_info = card.split()[6:8]
            usage = int(power_info[0].rstrip('W'))
            cap = int(power_info[2].rstrip('W'))
            if usage > cap * 0.9:  # Warning if using >90% of power cap
                print(f"Warning: Card {i} power usage is high: {usage}W/{cap}W")

            # Check memory usage
            mem_info = card.split()[11:14]
            used = int(mem_info[0].rstrip('MiB'))
            total = int(mem_info[2].rstrip('MiB'))
            if used > total * 0.9:  # Warning if using >90% of memory
                print(f"Warning: Card {i} memory usage is high: {used}MiB/{total}MiB")

        print(f"\nFound {card_count} Gaudi cards:")
        print(f"Driver Version: {output.split('Driver Version:')[1].split()[0]}")
        for i, card in enumerate(cards):
            temp = card.split()[2]
            power = card.split()[6:8]
            mem = card.split()[11:14]
            print(f"Card {i}: Temp: {temp}, Power: {power[0]}/{power[2]}, Memory: {mem[0]}/{mem[2]}")

        return True

    except Exception as e:
        print(f"Error checking Gaudi cards: {e}")
        return False


def check_system_requirements(required_cards: int = 1):
    """Check if system meets requirements"""
    try:
        # Check if docker is available
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            print("Error: Docker is not available")
            return False

        # Check if Habana runtime is available
        result = subprocess.run(["docker", "info"], capture_output=True, text=True)
        if "habana" not in result.stdout.lower():
            print("Error: Habana runtime not found in Docker")
            return False

        # Check disk space
        result = subprocess.run(["df", "-h", "/"], capture_output=True, text=True)
        if result.returncode == 0:
            # Parse available space (this is a simple check, might need adjustment)
            available = int(result.stdout.split("\n")[1].split()[3].rstrip("G"))
            if available < 100:  # Less than 100GB
                print("Warning: Low disk space available")
                return False

        # Check Gaudi cards
        if not check_gaudi_cards(required_cards):
            return False

        return True
    except Exception as e:
        print(f"Error checking system requirements: {e}")
        return False


def run_command(cmd):
    """Run a command with timeout"""
    try:
        process = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate(timeout=3600)  # 1 hour timeout
        return stdout.decode(), stderr.decode(), process.returncode
    except subprocess.TimeoutExpired:
        process.kill()
        print(f"Command timed out after 1 hour: {cmd}")
        return "", "Timeout", 1
    except Exception as e:
        print(f"Error running command: {e}")
        return "", str(e), 1


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
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Load configuration first to get max cards required
    config_path = "/workspace/benchmark_config.json"
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at {config_path}")
        return

    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in config file: {e}")
        return
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return

    # Get maximum number of cards required
    max_cards = max([model.get("cards_required", 1) for model in config.get("models", [])])

    # Check system requirements including Gaudi cards
    print("\nChecking system requirements...")
    if not check_system_requirements(max_cards):
        print("System requirements not met. Exiting.")
        return

    # Check for HuggingFace token
    if not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        print("Error: HUGGING_FACE_HUB_TOKEN environment variable is not set")
        print("Please set it in your environment before running the script:")
        print("export HUGGING_FACE_HUB_TOKEN='your-token-here'")
        return

    # Validate configuration
    if not validate_config(config):
        print("Invalid configuration. Exiting.")
        return

    # Sort models by size
    models = sorted(config["models"], key=lambda x: x["size"])
    
    # Pre-download and validate all models
    print("\nValidating model availability...")
    if not download_models(models):
        print("\nSome models failed to download/validate.")
        response = input("Do you want to continue with available models? (y/n): ")
        if response.lower() != 'y':
            print("Aborting benchmark run.")
            return
        print("Continuing with available models...")

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
