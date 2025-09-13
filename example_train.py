#!/usr/bin/env python3
"""
Example training script demonstrating how to use the transformer training script.

This shows various configurations for training the transformer model with different
hyperparameters and data sources.
"""

import subprocess
import sys
from pathlib import Path

def run_training_example():
    """Run a basic training example using TinyStories data"""

    # Check if data files exist
    data_dir = Path("data")
    train_data = data_dir / "TinyStoriesV2-GPT4-train.txt"
    val_data = data_dir / "TinyStoriesV2-GPT4-valid.txt"

    if not train_data.exists():
        print(f"Training data not found at {train_data}")
        print("Please download the data first using the commands in README.md")
        return

    # Basic training configuration
    cmd = [
        sys.executable, "-m", "cs336_basics.train",
        "--data_path", str(train_data),
        "--val_data_path", str(val_data),
        "--tokenizer_type", "tiktoken",
        "--tiktoken_encoding", "gpt2",

        # Small model for quick testing
        "--vocab_size", "50257",
        "--context_length", "256",
        "--d_model", "256",
        "--num_layers", "4",
        "--num_heads", "4",
        "--d_ff", "1024",

        # Training settings
        "--batch_size", "8",
        "--max_iters", "100",
        "--warmup_iters", "10",
        "--learning_rate", "1e-3",
        "--min_lr", "1e-4",
        "--weight_decay", "0.1",

        # Logging
        "--log_interval", "10",
        "--eval_interval", "50",
        "--eval_iters", "10",
        "--save_interval", "50",

        # Experiment name
        "--exp_name", "tinystories_small_test",
        "--wandb_project_name", "",  # Disable wandb for this example

        # Device
        "--device", "auto",

        # Random seed for reproducibility
        "--seed", "42"
    ]

    print("Running basic training example...")
    print("Command:", " ".join(cmd))
    print()

    try:
        subprocess.run(cmd, check=True)
        print("\nTraining completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\nTraining failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return False

    return True

def show_configuration_examples():
    """Show examples of different training configurations"""

    print("=== Training Configuration Examples ===\n")

    print("1. Small model for quick testing:")
    print("python -m cs336_basics.train \\")
    print("  --data_path data/TinyStoriesV2-GPT4-train.txt \\")
    print("  --val_data_path data/TinyStoriesV2-GPT4-valid.txt \\")
    print("  --d_model 256 --num_layers 4 --num_heads 4 --d_ff 1024 \\")
    print("  --batch_size 8 --max_iters 1000 --context_length 256")
    print()

    print("2. Medium model (GPT-2 small-like):")
    print("python -m cs336_basics.train \\")
    print("  --data_path data/owt_train.txt \\")
    print("  --val_data_path data/owt_valid.txt \\")
    print("  --d_model 768 --num_layers 12 --num_heads 12 --d_ff 3072 \\")
    print("  --batch_size 32 --max_iters 10000 --context_length 1024 \\")
    print("  --learning_rate 6e-4 --weight_decay 0.1")
    print()

    print("3. Large model with custom tokenizer:")
    print("python -m cs336_basics.train \\")
    print("  --data_path data/owt_train.txt \\")
    print("  --tokenizer_type bpe \\")
    print("  --vocab_file path/to/vocab.json \\")
    print("  --merges_file path/to/merges.txt \\")
    print("  --d_model 1024 --num_layers 24 --num_heads 16 --d_ff 4096 \\")
    print("  --batch_size 64 --max_iters 50000")
    print()

    print("4. Resume training from checkpoint:")
    print("python -m cs336_basics.train \\")
    print("  --data_path data/owt_train.txt \\")
    print("  --resume_from checkpoints/checkpoint_5000.pt \\")
    print("  --max_iters 10000")
    print()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "examples":
        show_configuration_examples()
    else:
        success = run_training_example()
        if success:
            print("\n" + "="*50)
            print("Example training completed!")
            print("To see more configuration examples, run:")
            print("python example_train.py examples")
