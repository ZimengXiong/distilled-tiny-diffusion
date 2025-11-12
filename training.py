"""
Training script with [PAD] and [EOS] token support
FIXED: Loads sample data for proper evaluation context
"""

import os
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from model import DiffusionTransformer, DiffusionConfig, encode_text, decode_tokens
from tokenizer_utils import get_tokenizer, vocab_size, mask_token_id

torch.serialization.add_safe_globals([DiffusionConfig])

class MaskedDiffusionSchedule:
    def __init__(self, num_timesteps, mask_token_id, pad_token_id, eos_token_id, context_len=0):
        self.num_timesteps = num_timesteps
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.context_len = context_len
        self.mask_probs = torch.linspace(1.0 / num_timesteps, 1.0, num_timesteps)

    def add_masks(self, x_0, t):
        B, T = x_0.shape
        device = x_0.device
        mask_prob = self.mask_probs[t.cpu()].to(device)
        mask = torch.rand(B, T, device=device) < mask_prob.unsqueeze(1)
        
        # Don't mask context
        if self.context_len > 0:
            mask[:, :self.context_len] = False
        
        # Don't mask [PAD] or [EOS] tokens
        is_special = (x_0 == self.pad_token_id) | (x_0 == self.eos_token_id)
        mask = mask & ~is_special
        
        mask_token = torch.full_like(x_0, self.mask_token_id)
        x_t = torch.where(mask, mask_token, x_0)
        return x_t

def get_data_loader(data_path, batch_size, seq_len, device):
    """Load padded samples - each line is one complete 128-token sample"""
    print(f"Loading padded samples from {data_path}...")
    
    samples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                # Parse space-separated token IDs
                token_ids = [int(t) for t in line.split()]
                if len(token_ids) == seq_len:
                    samples.append(torch.tensor(token_ids, dtype=torch.long))
                else:
                    print(f"Warning: Sample has {len(token_ids)} tokens, expected {seq_len}")
    
    print(f"Loaded {len(samples)} samples")
    
    if len(samples) == 0:
        raise ValueError("No valid samples found!")
    
    def data_generator():
        while True:
            indices = torch.randint(0, len(samples), (batch_size,))
            batch = torch.stack([samples[i] for i in indices]).to(device)
            yield batch
    
    return data_generator()

def train_step(model, x_0, mask_schedule, optimizer):
    B, _ = x_0.shape
    device = x_0.device
    t = torch.randint(0, mask_schedule.num_timesteps, (B,), device=device)
    x_t = mask_schedule.add_masks(x_0, t)
    logits = model(x_t, t)
    mask = (x_t == mask_schedule.mask_token_id).float()
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), x_0.view(-1), reduction="none")
    masked_loss = loss.view(B, -1) * mask
    mask_count = mask.sum()
    loss = masked_loss.sum() / mask_count.clamp(min=1)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item()

def extract_step_from_filename(filename):
    match = re.search(r'step_(\d+)', str(filename))
    if match:
        return int(match.group(1))
    return None

def train(model, data_loader, mask_schedule, optimizer, scheduler, num_steps=10000, 
          sample_interval=1000, dataset_tokens=None, start_step=0, checkpoint_prefix="diffusion_model"):
    model.train()
    os.makedirs("weights", exist_ok=True)
    
    pbar = tqdm(range(start_step, num_steps), desc="Training", initial=start_step, total=num_steps)
    for step in pbar:
        x_0 = next(data_loader)
        loss = train_step(model, x_0, mask_schedule, optimizer)
        scheduler.step()
        
        current_lr = scheduler.get_last_lr()[0]
        pbar.set_postfix({"loss": f"{loss:.4f}", "lr": f"{current_lr:.2e}"})
        
        if (step + 1) % sample_interval == 0:
            model.eval()
            with torch.no_grad():
                context_tokens = None
                if model.config.context_len > 0 and dataset_tokens is not None:
                    ctx_len = model.config.context_len
                    # Use first ctx_len tokens from dataset_tokens as context
                    if len(dataset_tokens) >= ctx_len:
                        context_tokens = dataset_tokens[:ctx_len].unsqueeze(0)
                
                samples = model.sample(
                    batch_size=1,
                    seq_len=model.config.sequence_len,
                    device=model.get_device(),
                    context_tokens=context_tokens,
                    method="confidence",
                    confidence_threshold=0.9,
                )
                text = decode_tokens(samples[0])
                tqdm.write(f"\n--- Sample at step {step + 1} ---")
                tqdm.write(text[:300])
                tqdm.write("---\n")
            
            checkpoint_path = f"weights/{checkpoint_prefix}_step_{step + 1}.pt"
            checkpoint = {
                'step': step + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': model.config,
            }
            torch.save(checkpoint, checkpoint_path)
            tqdm.write(f"Checkpoint saved: {checkpoint_path}")
            
            model.train()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train diffusion model with optional resume")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--steps", type=int, default=20000, help="Total training steps")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--eval-interval", type=int, default=1000, help="Sample and checkpoint interval")
    parser.add_argument("--data", type=str, default="data/alpaca_padded.txt", help="Training data path")
    parser.add_argument("--checkpoint-prefix", type=str, default="alpaca_expert", help="Prefix for checkpoint filenames")
    args = parser.parse_args()
    
    data_path = Path(args.data)
    
    # Initialize tokenizer
    print("Initializing tokenizer...")
    get_tokenizer(data_paths=[data_path])
    v_size = vocab_size()
    m_id = mask_token_id()
    print(f"Tokenizer ready. Vocab size: {v_size}, Mask ID: {m_id}\n")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else
                         ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}\n")
    
    # Resume or create new
    start_step = 0
    
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        
        if 'step' in checkpoint:
            start_step = checkpoint['step']
        else:
            step_from_name = extract_step_from_filename(args.resume)
            if step_from_name:
                start_step = step_from_name
        
        print(f"Resuming from step {start_step}")
        
        remaining_steps = args.steps - start_step
        if remaining_steps <= 0:
            print(f"Error: Already at step {start_step}, target is {args.steps}")
            return
        
        print(f"Will train for {remaining_steps} more steps ({start_step} → {args.steps})")
        
        if 'config' in checkpoint:
            config = checkpoint['config']
        else:
            config = DiffusionConfig(vocab_size=v_size, mask_token_id=m_id)
        
        model = DiffusionTransformer(config).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=remaining_steps, eta_min=args.lr * 0.01)
        
        if 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("Optimizer state restored")
            except Exception as e:
                print(f"Warning: Could not restore optimizer state: {e}")
                print("Starting with fresh optimizer")
        
        print(f"Model and optimizer restored\n")
        
    else:
        print("Starting fresh training...")
        config = DiffusionConfig(vocab_size=v_size, mask_token_id=m_id)
        print(f"Config: seq_len={config.sequence_len}, vocab={config.vocab_size}")
        
        model = DiffusionTransformer(config).to(device)
        model.init_weights()
        
        if device.type == "mps":
            print("Warming up MPS...")
            with torch.no_grad():
                dummy = torch.randint(0, config.vocab_size, (1, config.sequence_len), device=device)
                dummy_t = torch.tensor([0], device=device)
                _ = model(dummy, dummy_t)
            print("MPS ready!")
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr * 0.01)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,}\n")
    
    # Get special token IDs
    tokenizer = get_tokenizer()
    pad_token_id = tokenizer.token_to_id("[PAD]")
    eos_token_id = tokenizer.token_to_id("[EOS]")
    
    print(f"Special tokens - PAD: {pad_token_id}, EOS: {eos_token_id}\n")
    
    # Training setup
    mask_schedule = MaskedDiffusionSchedule(
        config.diffusion_steps, 
        config.mask_token_id, 
        pad_token_id,
        eos_token_id,
        config.context_len
    )
    
    # Data loader
    print("Creating data loader...")
    data_loader = get_data_loader(str(data_path), args.batch_size, config.sequence_len, device)
    print("Data loader ready!\n")
    
    # CRITICAL FIX: Load first sample for evaluation context
    print("Loading sample for evaluation context...")
    with open(data_path, 'r') as f:
        first_line = f.readline().strip()
        dataset_tokens = torch.tensor([int(t) for t in first_line.split()], dtype=torch.long)
    
    print(f"Evaluation context loaded: {len(dataset_tokens)} tokens")
    
    # Decode to show what we're using
    sample_text = decode_tokens(dataset_tokens[:64])
    print(f"Sample context: {sample_text[:100]}...\n")
    
    # Train
    print(f"Starting training from step {start_step} to {args.steps}...\n")
    train(model, data_loader, mask_schedule, optimizer, scheduler, args.steps, 
          args.eval_interval, dataset_tokens, start_step, args.checkpoint_prefix)
    
    # Final save
    final_path = f"weights/{args.checkpoint_prefix}_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"Training complete! Final model saved to {final_path}")

if __name__ == "__main__":
    main()
