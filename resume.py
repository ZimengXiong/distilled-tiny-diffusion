"""
resume_20k_to_100k.py - One-time script to continue training from 20k to 100k steps
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
from model import DiffusionTransformer, DiffusionConfig, encode_text, decode_tokens
from training import MaskedDiffusionSchedule

def get_data_loader(data_path, batch_size, sequence_len, device):
    """Infinite data loader"""
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    tokens = encode_text(text)
    dataset_size = len(tokens) - sequence_len
    
    while True:
        start_indices = torch.randint(0, dataset_size, (batch_size,))
        batch = torch.stack([
            tokens[start:start + sequence_len] 
            for start in start_indices
        ]).to(device)
        yield batch

def main():
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Load existing model from 20k checkpoint
    print("Loading model from weights/diffusion_model.pt...")
    config = DiffusionConfig()
    model = DiffusionTransformer(config).to(device)
    model.load_state_dict(torch.load("weights/diffusion_model.pt", map_location=device))
    model.train()
    print(f"Model loaded! Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Optimizer - create fresh (will lose momentum from first 20k, but acceptable)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-4,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )
    
    # Scheduler - configure for REMAINING 80k steps (20k → 100k)
    from torch.optim.lr_scheduler import CosineAnnealingLR
    remaining_steps = 80000  # 100k - 20k
    scheduler = CosineAnnealingLR(optimizer, T_max=remaining_steps, eta_min=1e-5)
    
    # Mask schedule
    mask_schedule = MaskedDiffusionSchedule(
        num_timesteps=config.diffusion_steps,
        mask_token_id=config.mask_token_id,
        context_len=config.context_len
    )
    
    # Data loader
    data_path = "data/tiny_shakespeare.txt"  # Change to your conversational data
    batch_size = 64
    data_loader = get_data_loader(data_path, batch_size, config.sequence_len, device)
    
    # Training loop: 20k → 100k
    start_step = 20000
    end_step = 100000
    
    print(f"Resuming training from step {start_step} to {end_step}...")
    print(f"Total remaining steps: {remaining_steps}\n")
    
    pbar = tqdm(range(start_step, end_step), desc="Training", initial=start_step, total=end_step)
    
    for step in pbar:
        # Get batch
        x_0 = next(data_loader)
        B, T = x_0.shape
        
        # Sample timesteps
        t = torch.randint(0, mask_schedule.num_timesteps, (B,), device=device)
        
        # Add masks
        x_t = mask_schedule.add_masks(x_0, t)
        
        # Forward pass
        logits = model(x_t, t)
        
        # Compute loss (cross-entropy on masked positions)
        mask = (x_t == mask_schedule.mask_token_id)
        loss = F.cross_entropy(
            logits[mask],
            x_0[mask],
            reduction='mean'
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        # Logging
        current_lr = scheduler.get_last_lr()[0]
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'lr': f'{current_lr:.2e}'
        })
        
        # Save checkpoint every 5000 steps
        if (step + 1) % 5000 == 0:
            torch.save(model.state_dict(), f"weights/diffusion_model_step_{step+1}.pt")
            print(f"\nCheckpoint saved at step {step+1}")
            
            # Generate sample to monitor quality
            model.eval()
            with torch.no_grad():
                samples = model.sample(
                    batch_size=1,
                    seq_len=config.sequence_len,
                    device=device,
                    method='confidence',
                    confidence_threshold=0.9
                )
                text = decode_tokens(samples[0])
                print(f"\n--- Sample at step {step+1} ---")
                print(text[:300])
                print("---\n")
            model.train()
    
    # Final save
    torch.save(model.state_dict(), "weights/diffusion_model.pt")
    print("\n✓ Training complete! Final model saved to weights/diffusion_model.pt")

if __name__ == "__main__":
    main()
