#!/usr/bin/env python3
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
import logging
import traceback

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("benchmark.log"), logging.StreamHandler(sys.stdout)],
)


def emergency_cleanup():
    """Emergency cleanup in case of critical failure"""
    logging.critical("Emergency cleanup triggered")
    try:
        # Save any pending results
        if "results" in globals() and results:
            with open("emergency_results.csv", "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
            logging.info("Emergency results saved to emergency_results.csv")
    except Exception as e:
        logging.error(f"Failed to save emergency results: {e}")


def signal_handler(signum, frame):
    """Handle termination signals gracefully"""
    logging.warning(f"Received signal {signum}")
    emergency_cleanup()
    sys.exit(1)


def validate_config(config: Dict) -> bool:
    """Validate the configuration file structure"""
    try:
        required_fields = ["models"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")

        if not config["models"]:
            raise ValueError("No models specified in configuration")

        for model in config["models"]:
            model_fields = ["name", "size", "cards_required", "configs"]
            for field in model_fields:
                if field not in model:
                    raise ValueError(
                        f"Missing required field '{field}' in model: {model['name']}"
                    )

            if not model["configs"]:
                raise ValueError(
                    f"No configurations specified for model: {model['name']}"
                )

            for cfg in model["configs"]:
                config_fields = ["input_tokens", "output_tokens", "batch_size"]
                for field in config_fields:
                    if field not in cfg:
                        raise ValueError(
                            f"Missing required field '{field}' in config for model: {model['name']}"
                        )

        return True

    except Exception as e:
        logging.error(f"Configuration validation failed: {str(e)}")
        logging.error(traceback.format_exc())
        return False


def check_gaudi_cards(required_cards: int) -> bool:
    """Check if required number of Gaudi cards are available and healthy"""
    try:
        # Check if hl-smi is available
        result = subprocess.run(["which", "hl-smi"], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError("hl-smi not found. Are Gaudi drivers installed?")

        # Get hl-smi output
        result = subprocess.run(["hl-smi"], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError("Error running hl-smi command")

        output = result.stdout

        # Check driver version
        if "Driver Version:" not in output:
            raise RuntimeError("Could not find Gaudi driver version")

        # Count available cards
        card_count = output.count("HL-225")
        if card_count < required_cards:
            raise RuntimeError(
                f"Not enough Gaudi cards. Required: {required_cards}, Found: {card_count}"
            )

        # Parse card details
        cards = []
        for line in output.split("\n"):
            if "HL-225" in line:
                cards.append(line)

        # Check each card's status
        for i, card in enumerate(cards):
            # Check temperature (example threshold: 85°C)
            temp = int(card.split()[2].rstrip("C"))
            if temp > 85:
                logging.warning(f"Card {i} temperature is high: {temp}°C")

            # Check power usage
            power_info = card.split()[6:8]
            usage = int(power_info[0].rstrip("W"))
            cap = int(power_info[2].rstrip("W"))
            if usage > cap * 0.9:  # Warning if using >90% of power cap
                logging.warning(f"Card {i} power usage is high: {usage}W/{cap}W")

            # Check memory usage
            mem_info = card.split()[11:14]
            used = int(mem_info[0].rstrip("MiB"))
            total = int(mem_info[2].rstrip("MiB"))
            if used > total * 0.9:  # Warning if using >90% of memory
                logging.warning(f"Card {i} memory usage is high: {used}MiB/{total}MiB")

        logging.info(f"Found {card_count} Gaudi cards:")
        logging.info(f"Driver Version: {output.split('Driver Version:')[1].split()[0]}")
        for i, card in enumerate(cards):
            temp = card.split()[2]
            power = card.split()[6:8]
            mem = card.split()[11:14]
            logging.info(
                f"Card {i}: Temp: {temp}, Power: {power[0]}/{power[2]}, Memory: {mem[0]}/{mem[2]}"
            )

        return True

    except Exception as e:
        logging.error(f"Gaudi card check failed: {str(e)}")
        logging.error(traceback.format_exc())
        return False


def check_system_requirements(required_cards: int = 1):
    """Check if system meets requirements"""
    try:
        # Check if docker is available
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError("Docker is not available")

        # Check if Habana runtime is available
        result = subprocess.run(["docker", "info"], capture_output=True, text=True)
        if "habana" not in result.stdout.lower():
            raise RuntimeError("Habana runtime not found in Docker")

        # Check disk space
        result = subprocess.run(["df", "-h", "/"], capture_output=True, text=True)
        if result.returncode == 0:
            # Parse available space (this is a simple check, might need adjustment)
            available = int(result.stdout.split("\n")[1].split()[3].rstrip("G"))
            if available < 100:  # Less than 100GB
                logging.warning("Low disk space available")

        # Check Gaudi cards
        if not check_gaudi_cards(required_cards):
            raise RuntimeError("Gaudi card validation failed")

        return True
    except Exception as e:
        logging.error(f"System requirements check failed: {str(e)}")
        logging.error(traceback.format_exc())
        return False


def download_models(models: List[Dict]) -> bool:
    """
    Pre-download all models to ensure availability and permissions.
    Returns True if all models are downloaded successfully, False otherwise.
    """
    logging.info("\nPre-downloading models...")
    success = True
    failed_models = []

    for model in models:
        model_name = model["name"]
        logging.info(f"\nDownloading {model_name}...")
        try:
            # Try to load tokenizer first as it's smaller
            logging.info(f"Downloading tokenizer for {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_auth_token=os.environ.get("HUGGING_FACE_HUB_TOKEN"),
                timeout=60,  # Add timeout
            )

            # Then try to load model config (without downloading weights)
            logging.info(f"Validating model config for {model_name}")
            config = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_auth_token=os.environ.get("HUGGING_FACE_HUB_TOKEN"),
                config_only=True,
                timeout=60,  # Add timeout
            )

            # Validate model size if available in config
            if hasattr(config, "num_parameters"):
                expected_size = model["size"] * 1e9  # Convert B to parameters
                actual_size = config.num_parameters
                if (
                    abs(expected_size - actual_size) / expected_size > 0.1
                ):  # 10% tolerance
                    logging.warning(f"Model size mismatch for {model_name}")
                    logging.warning(
                        f"Expected: {model['size']}B, Actual: {actual_size/1e9:.1f}B"
                    )

            logging.info(f"✓ Successfully validated {model_name}")

        except Exception as e:
            logging.error(f"✗ Error downloading {model_name}: {str(e)}")
            failed_models.append(model_name)
            success = False

    if failed_models:
        logging.info("\nFailed models:")
        for model in failed_models:
            logging.info(f"  - {model}")

    return success


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
        logging.error(f"Command timed out after 1 hour: {cmd}")
        return "", "Timeout", 1
    except Exception as e:
        logging.error(f"Error running command: {e}")
        return "", str(e), 1


def run_benchmark_bf16(
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    batch_size: int,
    cards_required: int,
) -> Optional[Dict]:
    """Run benchmark in BF16 mode without quantization"""
    logging.info(f"\nRunning BF16 benchmark for {model_name}")
    logging.info(
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

    logging.info(f"Running command: {cmd}")
    start_time = time.time()
    stdout, stderr, returncode = run_command(cmd)
    end_time = time.time()

    if returncode != 0:
        logging.error(f"Error during BF16 benchmark: {stderr}")
        logging.error(f"Command output: {stdout}")
        return None

    throughput = None
    for line in stdout.split("\n"):
        if "tokens/sec" in line:
            try:
                throughput = float(line.split(":")[1].strip())
            except Exception as e:
                logging.error(f"Error parsing throughput: {e}")
                logging.error(f"Line content: {line}")
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
    logging.info(f"\nQuantizing model: {model_name}")
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

    logging.info(f"Running quantization command: {cmd}")
    stdout, stderr, returncode = run_command(cmd)
    if returncode != 0:
        logging.error(f"Error during quantization: {stderr}")
        logging.error(f"Command output: {stdout}")
        return False
    return True


def run_benchmark_quantized(
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    batch_size: int,
    cards_required: int,
) -> Optional[Dict]:
    logging.info(f"\nRunning quantized benchmark for {model_name}")
    logging.info(
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

    logging.info(f"Running quantized benchmark command: {cmd}")
    start_time = time.time()
    stdout, stderr, returncode = run_command(cmd)
    end_time = time.time()

    if returncode != 0:
        logging.error(f"Error during quantized benchmark: {stderr}")
        logging.error(f"Command output: {stdout}")
        return None

    throughput = None
    for line in stdout.split("\n"):
        if "tokens/sec" in line:
            try:
                throughput = float(line.split(":")[1].strip())
            except Exception as e:
                logging.error(f"Error parsing throughput: {e}")
                logging.error(f"Line content: {line}")
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
    try:
        # Set up signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        logging.info("Starting benchmark run")

        # Load configuration first to get max cards required
        workspace_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(workspace_dir, "benchmark_config.json")

        logging.info(f"Looking for config file at: {config_path}")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at: {config_path}")

        try:
            with open(config_path, "r") as f:
                config = json.load(f)
                logging.info("Successfully loaded configuration file")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file: {e}")
        except Exception as e:
            raise Exception(f"Error reading config file: {str(e)}")

        # Validate configuration
        if not validate_config(config):
            raise ValueError("Configuration validation failed")

        # Get maximum number of cards required
        max_cards = max(
            [model.get("cards_required", 1) for model in config.get("models", [])]
        )

        # Check system requirements including Gaudi cards
        logging.info("\nChecking system requirements...")
        if not check_system_requirements(max_cards):
            raise RuntimeError("System requirements not met")

        # Check for HuggingFace token
        if not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
            raise EnvironmentError(
                "HUGGING_FACE_HUB_TOKEN environment variable is not set"
            )

        # Initialize results storage
        global results
        results = []

        # Sort models by size
        models = sorted(config["models"], key=lambda x: x["size"])

        # Pre-download and validate all models
        logging.info("\nValidating model availability...")
        if not download_models(models):
            logging.info("\nSome models failed to download/validate.")
            response = input("Do you want to continue with available models? (y/n): ")
            if response.lower() != "y":
                logging.info("Aborting benchmark run.")
                return
            logging.info("Continuing with available models...")

        # Prepare results file
        results_file = "benchmark_results.csv"
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
            logging.error(f"Error creating results file: {e}")
            return

        # Run benchmarks for each model
        for model in models:
            logging.info(f"\n{'='*50}")
            logging.info(f"Starting benchmarks for {model['name']} ({model['size']}B)")
            logging.info(f"{'='*50}")

            # Try quantization first
            quant_success = quantize_model(model["name"], model["cards_required"])

            if not quant_success:
                logging.info(
                    f"\nQuantization failed for {model['name']}, will run in BF16 mode only"
                )

            # For each configuration, run either quantized or BF16 benchmark
            for cfg in model["configs"]:
                result = None

                # Try quantized run if quantization was successful
                if quant_success:
                    logging.info("\nAttempting quantized run...")
                    result = run_benchmark_quantized(
                        model["name"],
                        cfg["input_tokens"],
                        cfg["output_tokens"],
                        cfg["batch_size"],
                        model["cards_required"],
                    )

                    if result:
                        logging.info("Quantized run successful!")
                    else:
                        logging.info("Quantized run failed, falling back to BF16...")

                # If either quantization failed or quantized run failed, try BF16
                if not quant_success or not result:
                    logging.info("\nRunning BF16 version...")
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
                        logging.error(f"Error writing results: {e}")

                    logging.info(f"\nResults for {model['name']}:")
                    logging.info(f"Mode: {result['mode']}")
                    logging.info(f"Input tokens: {cfg['input_tokens']}")
                    logging.info(f"Output tokens: {cfg['output_tokens']}")
                    logging.info(f"Batch size: {cfg['batch_size']}")
                    logging.info(f"Throughput: {result['throughput']}")
                    logging.info("-" * 50)
                else:
                    logging.info(
                        f"Failed to get results for {model['name']} with config: {cfg}"
                    )
                    logging.info(
                        "Both quantized and BF16 runs failed for this configuration"
                    )

        logging.info("Benchmark run completed successfully")
        return 0

    except Exception as e:
        logging.critical(f"Critical error in main: {str(e)}")
        logging.critical(traceback.format_exc())
        emergency_cleanup()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
