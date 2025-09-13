import torch
from .training.adamw import AdamW
from .training.utils import get_batch, cross_entropy_loss, cosine_lr_schedule, gradient_clipping, save_checkpoint, load_checkpoint
from .transformer_model.transformer import TransformerLM
from .bpe_tokenizer.tokenizer import Tokenizer
import time
import wandb
import argparse
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
from pathlib import Path
import tiktoken
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Transformer Language Model")

    # Experiment settings
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--exp_name", type=str, default="transformer_lm", help="Experiment name")
    parser.add_argument("--wandb_project_name", type=str, default="JingLM", help="Wandb project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Wandb entity")
    parser.add_argument("--torch_deterministic", type=bool, default=True, help="Set torch.backends.cudnn.deterministic")

    # Data settings
    parser.add_argument("--data_path", type=str, required=True, help="Path to training data file")
    parser.add_argument("--val_data_path", type=str, default=None, help="Path to validation data file")
    parser.add_argument("--tokenizer_type", type=str, choices=["bpe", "tiktoken"], default="tiktoken", help="Tokenizer type")
    parser.add_argument("--vocab_file", type=str, default=None, help="Path to vocab file for BPE tokenizer")
    parser.add_argument("--merges_file", type=str, default=None, help="Path to merges file for BPE tokenizer")
    parser.add_argument("--tiktoken_encoding", type=str, default="gpt2", help="Tiktoken encoding name")

    # Model hyperparameters
    parser.add_argument("--vocab_size", type=int, default=50257, help="Vocabulary size")
    parser.add_argument("--context_length", type=int, default=1024, help="Context length")
    parser.add_argument("--d_model", type=int, default=768, help="Model dimension")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=3072, help="Feed-forward dimension")
    parser.add_argument("--rope_theta", type=float, default=10000.0, help="RoPE theta parameter")

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=6e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--betas", type=float, nargs=2, default=[0.9, 0.95], help="AdamW beta parameters")
    parser.add_argument("--eps", type=float, default=1e-8, help="AdamW epsilon")
    parser.add_argument("--max_iters", type=int, default=10000, help="Maximum number of training iterations")
    parser.add_argument("--warmup_iters", type=int, default=1000, help="Number of warmup iterations")
    parser.add_argument("--min_lr", type=float, default=6e-5, help="Minimum learning rate for cosine schedule")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping norm")

    # Evaluation and logging
    parser.add_argument("--eval_interval", type=int, default=500, help="Evaluation interval")
    parser.add_argument("--eval_iters", type=int, default=100, help="Number of evaluation iterations")
    parser.add_argument("--log_interval", type=int, default=10, help="Logging interval")
    parser.add_argument("--save_interval", type=int, default=1000, help="Checkpoint saving interval")

    # Device settings
    parser.add_argument("--device", type=str, default="auto", help="Device to use (cpu, cuda, mps, auto)")

    # Checkpointing
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")

    args = parser.parse_args()
    return args

def setup_device(device_str: str) -> str:
    """Setup and return the appropriate device"""
    if device_str == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device_str

def load_tokenizer(args):
    """Load tokenizer based on configuration"""
    if args.tokenizer_type == "tiktoken":
        return tiktoken.get_encoding(args.tiktoken_encoding)
    elif args.tokenizer_type == "bpe":
        if not args.vocab_file or not args.merges_file:
            raise ValueError("BPE tokenizer requires --vocab_file and --merges_file")
        return Tokenizer.from_files(args.vocab_file, args.merges_file)
    else:
        raise ValueError(f"Unknown tokenizer type: {args.tokenizer_type}")

def load_dataset(data_path: str, tokenizer, max_tokens: int = None) -> np.ndarray:
    """Load and tokenize dataset"""
    print(f"Loading dataset from {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()

    print("Tokenizing text...")
    tokens = tokenizer.encode(text)

    if max_tokens and len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        print(f"Truncated dataset to {max_tokens} tokens")

    print(f"Dataset loaded with {len(tokens)} tokens")
    return np.array(tokens, dtype=np.int32)

@torch.no_grad()
def evaluate_model(model, eval_dataset, args, device):
    """Evaluate model on validation dataset"""
    model.eval()
    total_loss = 0.0

    for _ in range(args.eval_iters):
        inputs, targets = get_batch(
            eval_dataset,
            args.batch_size,
            args.context_length,
            device
        )

        logits = model(inputs)
        # Reshape for cross entropy: (batch_size * seq_len, vocab_size) and (batch_size * seq_len,)
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)

        loss = cross_entropy_loss(logits_flat, targets_flat)
        total_loss += loss.item()

    model.train()
    return total_loss / args.eval_iters

def train():
    args = parse_args()

    # Setup device
    device = setup_device(args.device)
    print(f"Using device: {device}")

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Setup experiment name and logging
    run_name = f"{args.exp_name}_{int(time.time())}"

    # Initialize wandb
    if args.wandb_project_name:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            sync_tensorboard=True,
            name=run_name,
            save_code=True,
        )

    # Setup TensorBoard
    writer = SummaryWriter(f"runs/{run_name}")

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Load tokenizer
    tokenizer = load_tokenizer(args)

    # Load datasets
    train_dataset = load_dataset(args.data_path, tokenizer)
    eval_dataset = None
    if args.val_data_path:
        eval_dataset = load_dataset(args.val_data_path, tokenizer)

    # Initialize model
    print("Initializing model...")
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    ).to(device)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=args.betas,
        eps=args.eps
    )

    # Resume from checkpoint if specified
    start_iter = 0
    if args.resume_from:
        print(f"Resuming from checkpoint: {args.resume_from}")
        start_iter = load_checkpoint(args.resume_from, model, optimizer)
        print(f"Resumed from iteration {start_iter}")

    # Training loop
    print("Starting training...")
    model.train()

    for iter_num in tqdm(range(start_iter, args.max_iters), initial=start_iter, total=args.max_iters):
        # Learning rate schedule
        if iter_num < args.warmup_iters:
            lr = args.learning_rate * (iter_num + 1) / args.warmup_iters
        else:
            lr = cosine_lr_schedule(
                t=iter_num,
                max_lr=args.learning_rate,
                min_lr=args.min_lr,
                t_warmup=args.warmup_iters,
                t_cosine=args.max_iters
            )

        # Update learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Get batch
        inputs, targets = get_batch(train_dataset, args.batch_size, args.context_length, device)

        # Forward pass
        optimizer.zero_grad()
        logits = model(inputs)

        # Reshape for cross entropy
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)

        loss = cross_entropy_loss(logits_flat, targets_flat)

        # Backward pass
        loss.backward()

        # Gradient clipping
        if args.grad_clip > 0:
            gradient_clipping(model.parameters(), args.grad_clip)

        optimizer.step()

        # Logging
        if iter_num % args.log_interval == 0:
            print(f"iter {iter_num}: loss {loss.item():.4f}, lr {lr:.2e}")

            # Log to tensorboard and wandb
            writer.add_scalar("train/loss", loss.item(), iter_num)
            writer.add_scalar("train/learning_rate", lr, iter_num)

            if args.wandb_project_name:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/learning_rate": lr,
                    "iter": iter_num
                })

        # Evaluation
        if eval_dataset is not None and iter_num % args.eval_interval == 0 and iter_num > 0:
            eval_loss = evaluate_model(model, eval_dataset, args, device)
            print(f"iter {iter_num}: eval loss {eval_loss:.4f}")

            writer.add_scalar("eval/loss", eval_loss, iter_num)
            if args.wandb_project_name:
                wandb.log({"eval/loss": eval_loss, "iter": iter_num})

        # Save checkpoint
        if iter_num % args.save_interval == 0 and iter_num > 0:
            checkpoint_path = save_dir / f"checkpoint_{iter_num}.pt"
            save_checkpoint(model, optimizer, iter_num, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    # Save final checkpoint
    final_checkpoint_path = save_dir / f"final_checkpoint.pt"
    save_checkpoint(model, optimizer, args.max_iters, final_checkpoint_path)
    print(f"Saved final checkpoint to {final_checkpoint_path}")

    # Close logging
    writer.close()
    if args.wandb_project_name:
        wandb.finish()

    print("Training completed!")

if __name__ == "__main__":
    train()
