# Transformer Training Guide

This guide explains how to use the complete training script (`cs336_basics/train.py`) to train transformer language models with configurable hyperparameters.

## Features

The training script provides comprehensive functionality for training transformer models:

- **Flexible Model Architecture**: Configure model size, layers, attention heads, and more
- **Multiple Tokenizers**: Support for both tiktoken and BPE tokenizers
- **Advanced Training**: AdamW optimizer, cosine learning rate schedule, gradient clipping
- **Monitoring**: TensorBoard and Weights & Biases integration
- **Checkpointing**: Save and resume training from checkpoints
- **Evaluation**: Automatic validation on held-out data

## Quick Start

### Basic Usage

```bash
python3 -m cs336_basics.train \
  --data_path data/TinyStoriesV2-GPT4-train.txt \
  --val_data_path data/TinyStoriesV2-GPT4-valid.txt \
  --max_iters 1000 \
  --batch_size 32
```

### Key Arguments

**Required:**
- `--data_path`: Path to training data file (required)

**Model Configuration:**
- `--vocab_size`: Vocabulary size (default: 50257)
- `--context_length`: Maximum sequence length (default: 1024)
- `--d_model`: Model dimension (default: 768)
- `--num_layers`: Number of transformer layers (default: 12)
- `--num_heads`: Number of attention heads (default: 12)
- `--d_ff`: Feed-forward dimension (default: 3072)
- `--rope_theta`: RoPE positional encoding parameter (default: 10000.0)

**Training Configuration:**
- `--batch_size`: Batch size (default: 32)
- `--learning_rate`: Initial learning rate (default: 6e-4)
- `--weight_decay`: Weight decay for AdamW (default: 0.1)
- `--max_iters`: Maximum training iterations (default: 10000)
- `--warmup_iters`: Learning rate warmup iterations (default: 1000)
- `--grad_clip`: Gradient clipping norm (default: 1.0)

## Configuration Examples

### 1. Small Model for Testing

Quick training on a small model:

```bash
python3 -m cs336_basics.train \
  --data_path data/TinyStoriesV2-GPT4-train.txt \
  --val_data_path data/TinyStoriesV2-GPT4-valid.txt \
  --d_model 256 \
  --num_layers 4 \
  --num_heads 4 \
  --d_ff 1024 \
  --context_length 256 \
  --batch_size 8 \
  --max_iters 1000 \
  --exp_name "small_test"
```

### 2. GPT-2 Small Configuration

Standard GPT-2 Small architecture:

```bash
python3 -m cs336_basics.train \
  --data_path data/owt_train.txt \
  --val_data_path data/owt_valid.txt \
  --vocab_size 50257 \
  --context_length 1024 \
  --d_model 768 \
  --num_layers 12 \
  --num_heads 12 \
  --d_ff 3072 \
  --batch_size 32 \
  --learning_rate 6e-4 \
  --max_iters 10000 \
  --exp_name "gpt2_small"
```

### 3. Larger Model

Scaled-up configuration:

```bash
python3 -m cs336_basics.train \
  --data_path data/owt_train.txt \
  --val_data_path data/owt_valid.txt \
  --d_model 1024 \
  --num_layers 24 \
  --num_heads 16 \
  --d_ff 4096 \
  --context_length 2048 \
  --batch_size 16 \
  --learning_rate 3e-4 \
  --max_iters 50000 \
  --exp_name "large_model"
```

### 4. Custom BPE Tokenizer

Using a custom BPE tokenizer:

```bash
python3 -m cs336_basics.train \
  --data_path data/custom_data.txt \
  --tokenizer_type bpe \
  --vocab_file path/to/vocab.json \
  --merges_file path/to/merges.txt \
  --vocab_size 32000 \
  --max_iters 20000
```

### 5. Resume Training

Resume from a checkpoint:

```bash
python3 -m cs336_basics.train \
  --data_path data/owt_train.txt \
  --resume_from checkpoints/checkpoint_5000.pt \
  --max_iters 10000
```

## Tokenizer Options

### Tiktoken (Default)

Uses OpenAI's tiktoken library:

```bash
--tokenizer_type tiktoken \
--tiktoken_encoding gpt2  # or cl100k_base, etc.
```

### BPE Tokenizer

Uses the custom BPE tokenizer implementation:

```bash
--tokenizer_type bpe \
--vocab_file path/to/vocab.json \
--merges_file path/to/merges.txt
```

## Training Features

### Learning Rate Schedule

The script uses a cosine learning rate schedule with warmup:

1. **Warmup phase**: Linear increase from 0 to `--learning_rate` over `--warmup_iters`
2. **Cosine decay**: Cosine decay from `--learning_rate` to `--min_lr`

### Optimizer Settings

AdamW optimizer with configurable parameters:

```bash
--learning_rate 6e-4    # Initial learning rate
--weight_decay 0.1      # Weight decay coefficient
--betas 0.9 0.95       # Beta1 and Beta2 parameters
--eps 1e-8             # Epsilon for numerical stability
```

### Gradient Clipping

Global gradient norm clipping:

```bash
--grad_clip 1.0  # Maximum gradient norm
```

## Monitoring and Logging

### TensorBoard

Automatically logs to TensorBoard:

```bash
tensorboard --logdir runs/
```

### Weights & Biases

Enable wandb logging:

```bash
--wandb_project_name "my_project" \
--wandb_entity "my_team"
```

### Logging Intervals

Control logging frequency:

```bash
--log_interval 10        # Log training metrics every N iterations
--eval_interval 500      # Evaluate on validation set every N iterations
--eval_iters 100        # Number of evaluation batches
--save_interval 1000    # Save checkpoint every N iterations
```

## Checkpointing

### Automatic Saving

Checkpoints are automatically saved at:
- Regular intervals (controlled by `--save_interval`)
- End of training (as `final_checkpoint.pt`)

### Resume Training

Resume from any checkpoint:

```bash
--resume_from checkpoints/checkpoint_5000.pt
```

The script will:
- Load model and optimizer state
- Resume from the correct iteration
- Continue with the same hyperparameters

### Custom Save Directory

Specify where to save checkpoints:

```bash
--save_dir ./my_checkpoints/
```

## Device Selection

### Automatic Device Selection

```bash
--device auto  # Automatically selects CUDA > MPS > CPU
```

### Manual Device Selection

```bash
--device cuda  # Force CUDA
--device mps   # Force Apple Silicon GPU
--device cpu   # Force CPU
```

## Advanced Usage

### Memory Optimization

For large models, use smaller batch sizes and gradient accumulation:

```bash
--batch_size 8          # Reduce batch size
--grad_clip 1.0        # Keep gradient clipping
```

### Reproducibility

Set seed for reproducible results:

```bash
--seed 42
--torch_deterministic true
```

### Custom Experiment Naming

```bash
--exp_name "my_experiment_v1"
```

## Validation and Evaluation

The script automatically:
- Evaluates on validation data at regular intervals
- Computes cross-entropy loss on held-out data
- Logs validation metrics to TensorBoard and wandb

Configure evaluation:

```bash
--val_data_path data/validation.txt  # Validation dataset
--eval_interval 500                   # Evaluate every 500 iterations
--eval_iters 100                     # Use 100 batches for evaluation
```

## Troubleshooting

### Memory Issues

If you run out of memory:

1. Reduce batch size: `--batch_size 8`
2. Reduce model size: `--d_model 512 --num_layers 8`
3. Reduce context length: `--context_length 512`

### Slow Training

If training is too slow:

1. Increase batch size (if memory allows): `--batch_size 64`
2. Use GPU: `--device cuda` or `--device mps`
3. Reduce evaluation frequency: `--eval_interval 1000`

### Divergent Loss

If loss becomes NaN or explodes:

1. Reduce learning rate: `--learning_rate 3e-4`
2. Increase warmup: `--warmup_iters 2000`
3. Reduce gradient clip: `--grad_clip 0.5`

## Example Training Commands

Here are some ready-to-use training commands for different scenarios:

### Quick Test (5 minutes)
```bash
python3 -m cs336_basics.train \
  --data_path data/TinyStoriesV2-GPT4-train.txt \
  --d_model 128 --num_layers 2 --num_heads 2 --d_ff 512 \
  --context_length 128 --batch_size 4 --max_iters 100 \
  --exp_name "quick_test"
```

### Short Training (1 hour)
```bash
python3 -m cs336_basics.train \
  --data_path data/TinyStoriesV2-GPT4-train.txt \
  --val_data_path data/TinyStoriesV2-GPT4-valid.txt \
  --d_model 256 --num_layers 4 --num_heads 4 --d_ff 1024 \
  --batch_size 16 --max_iters 2000 \
  --exp_name "tinystories_small"
```

### Full Training (overnight)
```bash
python3 -m cs336_basics.train \
  --data_path data/owt_train.txt \
  --val_data_path data/owt_valid.txt \
  --d_model 768 --num_layers 12 --num_heads 12 --d_ff 3072 \
  --batch_size 32 --max_iters 20000 \
  --wandb_project_name "transformer_training" \
  --exp_name "gpt2_small_owt"
```

## Output Files

The training script creates:

```
checkpoints/
├── checkpoint_1000.pt    # Regular checkpoints
├── checkpoint_2000.pt
├── ...
└── final_checkpoint.pt   # Final model

runs/
└── experiment_name_timestamp/  # TensorBoard logs
    ├── events.out.tfevents...
    └── ...
```

Each checkpoint contains:
- Model state dict
- Optimizer state dict
- Current iteration number
- All hyperparameters (via wandb config)
