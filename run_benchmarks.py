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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("benchmark.log"), logging.StreamHandler(sys.stdout)],
)


def emergency_cleanup():
    logging.critical("Emergency cleanup triggered")
    try:
        if "results" in globals() and results:
            with open("emergency_results.csv", "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
            logging.info("Emergency results saved to emergency_results.csv")
    except Exception as e:
        logging.error(f"Failed to save emergency results: {e}")


def signal_handler(signum, frame):
    logging.warning(f"Received signal {signum}")
    emergency_cleanup()
    sys.exit(1)


def validate_config(config: Dict) -> bool:
    try:
        if "models" not in config or not config["models"]:
            raise ValueError("No models specified in configuration")

        for model in config["models"]:
            model_fields = ["model_id", "size", "cards_required", "configs"]
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


def check_gaudi_cards(required_cards=8):
    try:
        result = subprocess.run(["hl-smi"], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"hl-smi command failed: {result.stderr}")

        card_count = result.stdout.count("HL-225")
        logging.info(f"Found {card_count} Gaudi cards")

        if card_count < required_cards:
            raise RuntimeError(f"Found only {card_count} cards, need {required_cards}")

        return True

    except FileNotFoundError:
        raise RuntimeError("hl-smi command not found. Are Habana drivers installed?")
    except Exception as e:
        logging.error(f"Gaudi card check failed: {str(e)}")
        logging.error(traceback.format_exc())
        return False


def check_system_requirements(required_cards=1):
    try:
        in_container = os.path.exists("/.dockerenv")

        if not in_container:
            logging.info("Checking Docker installation...")
            result = subprocess.run(
                ["docker", "--version"], capture_output=True, text=True
            )
            if result.returncode != 0:
                raise RuntimeError("Docker is not installed or not in PATH")
            logging.info(f"Found Docker: {result.stdout.strip()}")

        if not check_gaudi_cards(required_cards):
            raise RuntimeError(f"Required {required_cards} Gaudi cards not available")

        min_space_gb = 100
        workspace_dir = os.path.dirname(os.path.abspath(__file__))
        statvfs = os.statvfs(workspace_dir)
        free_space_gb = (statvfs.f_frsize * statvfs.f_bavail) / (1024**3)
        if free_space_gb < min_space_gb:
            raise RuntimeError(
                f"Insufficient disk space. Required: {min_space_gb}GB, Available: {free_space_gb:.1f}GB"
            )
        logging.info(f"Disk space check passed. Available: {free_space_gb:.1f}GB")

        return True
    except Exception as e:
        logging.error(f"System requirements check failed: {str(e)}")
        logging.error(traceback.format_exc())
        return False


def download_models(models: List[Dict]) -> bool:
    logging.info("\nPre-downloading models...")
    success = True
    failed_models = []

    for model in models:
        model_id = model["model_id"]
        logging.info(f"\nDownloading {model_id}...")
        try:
            logging.info(f"Downloading {model_id}")
            AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
                use_auth_token=os.environ.get("HUGGING_FACE_HUB_TOKEN"),
            )
            AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                use_auth_token=os.environ.get("HUGGING_FACE_HUB_TOKEN"),
            )
            logging.info(f"✓ Successfully downloaded {model_id}")
        except Exception as e:
            logging.error(f"✗ Error downloading {model_id}: {str(e)}")
            failed_models.append(model_id)
            success = False

    if failed_models:
        logging.info("\nFailed models:")
        for model in failed_models:
            logging.info(f"  - {model}")
    return success


def run_command(cmd):
    try:
        process = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        stdout, stderr = process.communicate()
        return stdout, stderr, process.returncode
    except Exception as e:
        logging.error(f"Command execution failed: {e}")
        return "", str(e), 1


def run_benchmark_bf16(
    model_id: str,
    input_tokens: int,
    output_tokens: int,
    batch_size: int,
    cards_required: int,
) -> Optional[Dict]:
    logging.info(f"\nRunning BF16 benchmark for {model_id}")
    logging.info(
        f"Input tokens: {input_tokens}, Output tokens: {output_tokens}, Batch size: {batch_size}"
    )

    base_cmd = (
        f"python /workspace/optimum-habana/examples/gaudi_spawn.py --use_deepspeed --world_size {cards_required}"
        if cards_required > 1
        else "python"
    )
    cmd = f"""HF_DATASETS_TRUST_REMOTE_CODE=true QUANT_CONFIG=/workspace/optimum-habana/examples/text-generation/quantization_config/maxabs_quant.json TQDM_DISABLE=1 {base_cmd} \
    /workspace/optimum-habana/examples/text-generation/run_generation.py --model_name_or_path {model_id} \
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
        "model": model_id,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "batch_size": batch_size,
        "throughput": throughput,
        "time": end_time - start_time,
        "mode": "bf16",
    }


def quantize_model(model_id: str, cards_required: int) -> bool:
    logging.info(f"\nQuantizing model: {model_id}")
    base_cmd = (
        f"python /workspace/optimum-habana/examples/gaudi_spawn.py --use_deepspeed --world_size {cards_required}"
        if cards_required > 1
        else "python"
    )

    cmd = f"""HF_DATASETS_TRUST_REMOTE_CODE=true QUANT_CONFIG=/workspace/optimum-habana/examples/text-generation/quantization_config/maxabs_quant.json \
    TQDM_DISABLE=1 {base_cmd} \
    /workspace/optimum-habana/examples/text-generation/run_lm_eval.py --model_name_or_path {model_id} \
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
    -o quant_measure_{model_id.split('/')[-1]}.txt 2>&1 | tee -a /home/log_measur_quant.txt"""

    logging.info(f"Running quantization command: {cmd}")
    stdout, stderr, returncode = run_command(cmd)
    if returncode != 0:
        logging.error(f"Error during quantization: {stderr}")
        logging.error(f"Command output: {stdout}")
        return False
    return True


def run_benchmark_quantized(
    model_id: str,
    input_tokens: int,
    output_tokens: int,
    batch_size: int,
    cards_required: int,
) -> Optional[Dict]:
    logging.info(f"\nRunning quantized benchmark for {model_id}")
    logging.info(
        f"Input tokens: {input_tokens}, Output tokens: {output_tokens}, Batch size: {batch_size}"
    )

    base_cmd = (
        f"python /workspace/optimum-habana/examples/gaudi_spawn.py --use_deepspeed --world_size {cards_required}"
        if cards_required > 1
        else "python"
    )
    cmd = f"""HF_DATASETS_TRUST_REMOTE_CODE=true QUANT_CONFIG=/workspace/optimum-habana/examples/text-generation/quantization_config/maxabs_quant.json \
    TQDM_DISABLE=1 {base_cmd} /workspace/optimum-habana/examples/text-generation/run_generation.py --model_name_or_path {model_id} \
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
        "model": model_id,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "batch_size": batch_size,
        "throughput": throughput,
        "time": end_time - start_time,
        "mode": "quantized",
    }


def main():
    try:
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        logging.info("Starting benchmark run")

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

        if not validate_config(config):
            raise ValueError("Configuration validation failed")

        max_cards = max(
            [model.get("cards_required", 1) for model in config.get("models", [])]
        )

        logging.info("\nChecking system requirements...")
        if not check_system_requirements(max_cards):
            raise RuntimeError("System requirements not met")

        if not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
            raise EnvironmentError(
                "HUGGING_FACE_HUB_TOKEN environment variable is not set"
            )

        global results
        results = []

        models = sorted(config["models"], key=lambda x: x["size"])

        results_file = "benchmark_results.csv"
        fieldnames = [
            "timestamp",
            "model",
            "mode",
            "input_tokens",
            "output_tokens",
            "batch_size",
            "throughput",
            "time",
        ]

        if os.path.exists(results_file):
            logging.info(f"Appending to existing results file: {results_file}")
        else:
            with open(results_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

        for model in models:
            logging.info(f"\n{'='*50}")
            logging.info(f"Starting benchmarks for {model['name']} ({model['size']}B)")
            logging.info(f"{'='*50}")

            quant_success = quantize_model(model["model_id"], model["cards_required"])

            if not quant_success:
                logging.info(
                    f"\nQuantization failed for {model['model_id']}, will run in BF16 mode only"
                )

            for cfg in model["configs"]:
                if quant_success:
                    logging.info("\nAttempting quantized run...")
                    result = run_benchmark_quantized(
                        model["model_id"],
                        cfg["input_tokens"],
                        cfg["output_tokens"],
                        cfg["batch_size"],
                        model["cards_required"],
                    )

                if not quant_success or not result:
                    logging.info("\nRunning BF16 version...")
                    result = run_benchmark_bf16(
                        model["model_id"],
                        cfg["input_tokens"],
                        cfg["output_tokens"],
                        cfg["batch_size"],
                        model["cards_required"],
                    )

                if result:
                    result["timestamp"] = datetime.now().isoformat()
                    results.append(result)
                    try:
                        with open(results_file, "a", newline="") as f:
                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            writer.writerow(result)
                    except Exception as e:
                        logging.error(f"Error writing results: {e}")

                    logging.info(f"\nResults for {model['model_id']}:")
                    logging.info(f"Mode: {result['mode']}")
                    logging.info(f"Input tokens: {cfg['input_tokens']}")
                    logging.info(f"Output tokens: {cfg['output_tokens']}")
                    logging.info(f"Batch size: {cfg['batch_size']}")
                    logging.info(f"Throughput: {result['throughput']:.2f} tokens/sec")
                    logging.info(f"Time: {result['time']:.2f} seconds")
                    logging.info("-" * 50)
                else:
                    logging.info(
                        f"Failed to get results for {model['model_id']} with config: {cfg}"
                    )
                    logging.info(
                        "Both quantized and BF16 runs failed for this configuration"
                    )

        return 0

    except KeyboardInterrupt:
        logging.warning("\nBenchmark interrupted by user")
        emergency_cleanup()
        return 1
    except Exception as e:
        logging.error(f"Benchmark failed: {str(e)}")
        logging.error(traceback.format_exc())
        emergency_cleanup()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
