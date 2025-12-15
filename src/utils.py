"""
Utility functions for the RAG analysis project.
"""

import os
import json
import yaml
import random
import numpy as np
import torch
from typing import Dict, Any
import logging
from datetime import datetime


def setup_logging(log_dir: str = "logs", level: str = "INFO") -> logging.Logger:
    """
    Setup logging configuration.

    Args:
        log_dir: Directory to save log files
        level: Logging level

    Returns:
        Logger instance
    """
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"rag_experiment_{timestamp}.log")

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")

    return logger


def set_random_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"Random seed set to {seed}")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print(f"Loaded configuration from {config_path}")
    return config


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        output_path: Path to save config
    """
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Saved configuration to {output_path}")


def convert_numpy_types(obj: Any) -> Any:
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj


def save_json(data: Any, filepath: str, indent: int = 2) -> None:
    """
    Save data to JSON file.

    Args:
        data: Data to save
        filepath: Output file path
        indent: JSON indentation
    """
    # Convert numpy types to native Python types
    converted_data = convert_numpy_types(data)
    
    with open(filepath, "w") as f:
        json.dump(converted_data, f, indent=indent)

    print(f"Saved data to {filepath}")


def load_json(filepath: str) -> Any:
    """
    Load data from JSON file.

    Args:
        filepath: Input file path

    Returns:
        Loaded data
    """
    with open(filepath, "r") as f:
        data = json.load(f)

    print(f"Loaded data from {filepath}")
    return data


def format_prompt_template(template: str, **kwargs) -> str:
    """
    Format a prompt template with given arguments.

    Args:
        template: Template string with {placeholders}
        **kwargs: Values to fill in

    Returns:
        Formatted string
    """
    return template.format(**kwargs)


def get_device() -> str:
    """
    Get the appropriate device (cuda/mps/cpu).

    Returns:
        Device string
    """
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("Using Apple MPS device")
    else:
        device = "cpu"
        print("Using CPU device")

    return device


def estimate_memory_usage(model_name: str, precision: str = "fp16") -> float:
    """
    Estimate GPU memory usage for a model.

    Args:
        model_name: Model identifier
        precision: Precision type (fp32, fp16, int8)

    Returns:
        Estimated memory in GB
    """
    # Extract parameter count from model name (approximate)
    if "7b" in model_name.lower():
        params = 7e9
    elif "8b" in model_name.lower():
        params = 8e9
    elif "13b" in model_name.lower():
        params = 13e9
    else:
        params = 7e9  # Default assumption

    # Bytes per parameter based on precision
    bytes_per_param = {"fp32": 4, "fp16": 2, "int8": 1}

    memory_bytes = params * bytes_per_param.get(precision, 2)
    memory_gb = memory_bytes / (1024**3)

    # Add overhead (activations, gradients, etc.)
    memory_gb *= 1.2

    return memory_gb


def create_experiment_dir(base_dir: str, experiment_name: str = None) -> str:
    """
    Create a timestamped experiment directory.

    Args:
        base_dir: Base directory for experiments
        experiment_name: Optional experiment name

    Returns:
        Path to created directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if experiment_name:
        dir_name = f"{experiment_name}_{timestamp}"
    else:
        dir_name = f"experiment_{timestamp}"

    experiment_dir = os.path.join(base_dir, dir_name)
    os.makedirs(experiment_dir, exist_ok=True)

    # Create subdirectories
    os.makedirs(os.path.join(experiment_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "checkpoints"), exist_ok=True)

    print(f"Created experiment directory: {experiment_dir}")
    return experiment_dir


def print_results_table(results: Dict[str, Dict]) -> None:
    """
    Pretty print results table.

    Args:
        results: Dictionary of condition -> metrics
    """
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    # Header
    print(f"{'Condition':<20} {'EM':<10} {'F1':<10} {'Hall%':<10} {'Recall@k':<10}")
    print("-" * 80)

    # Rows
    for condition, metrics in results.items():
        em = f"{metrics.get('exact_match', 0):.3f}"
        f1 = f"{metrics.get('token_f1', 0):.3f}"
        hall = f"{metrics.get('hallucination_rate', 0):.3f}"
        recall = (
            f"{metrics.get('avg_recall@k', 0):.3f}"
            if "avg_recall@k" in metrics
            else "N/A"
        )

        print(f"{condition:<20} {em:<10} {f1:<10} {hall:<10} {recall:<10}")

    print("=" * 80 + "\n")


def validate_dataset_subset(
    subset: list, min_size: int = 50, max_size: int = 100
) -> bool:
    """
    Validate that dataset subset meets requirements.

    Args:
        subset: Dataset subset
        min_size: Minimum required size
        max_size: Maximum allowed size

    Returns:
        True if valid
    """
    if not subset:
        print("Error: Empty subset")
        return False

    if len(subset) < min_size:
        print(f"Warning: Subset size ({len(subset)}) below minimum ({min_size})")
        return False

    if len(subset) > max_size:
        print(f"Warning: Subset size ({len(subset)}) above maximum ({max_size})")
        return False

    # Check required fields
    required_fields = ["id", "question", "answer"]
    for i, example in enumerate(subset[:5]):  # Check first 5
        for field in required_fields:
            if field not in example:
                print(f"Error: Missing field '{field}' in example {i}")
                return False

    print(f"Subset validation passed ({len(subset)} examples)")
    return True


if __name__ == "__main__":
    # Example usage
    print("Utility functions module")
    print(f"Available device: {get_device()}")
