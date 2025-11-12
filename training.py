"""
Training script for discrete diffusion model with a BPE tokenizer.
"""

import os
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from model import DiffusionTransformer, DiffusionConfig, encode_text, decode_tokens
from tokenizer_utils import get_tokenizer, vocab_size, mask_token_id

class MaskedDiffusionSchedule:
    def __init__(self, num_timesteps, mask_token_id, context_len=0):
        self.num_timesteps = num_timesteps
        self.mask_token_id = mask_token_id
        self.context_len = context_len
        self.mask_probs = torch.linspace(1.0 / num_timesteps, 1.0, num_timesteps)

    def add_masks(self, x_0, t):
        B, T = x_0.shape
        device = x_0.device
        mask_prob = self.mask_probs[t.cpu()].to(device)
        mask = torch.rand(B, T, device=device) < mask_prob.unsqueeze(1)
        if self.context_len > 0:
            mask[:, :self.context_len] = False
        mask_token = torch.full_like(x_0, self.mask_token_id)
        x_t = torch.where(mask, mask_token, x_0)
        return x_t

def get_data_loader(data_path, batch_size, seq_len, device):
    print(f"Loading and tokenizing {data_path}...")
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"Text loaded: {len(text)} characters")
    tokens = encode_text(text)
    print(f"Tokenized: {len(tokens)} tokens")
    dataset_size = len(tokens) - seq_len
    if dataset_size <= 0:
        raise ValueError(f"Dataset too small: {len(tokens)} tokens < {seq_len} seq_len")
    
    def data_generator():
        while True:
            start_indices = torch.randint(0, dataset_size, (batch_size,))
            batch = torch.stack([tokens[start:start + seq_len] for start in start_indices]).to(device)
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

def train(model, data_loader, mask_schedule, optimizer, num_steps=10000, sample_interval=1000, dataset_tokens=None):
    model.train()
    os.makedirs("weights", exist_ok=True)
    pbar = tqdm(range(num_steps), desc="Training")
    for step in pbar:
        x_0 = next(data_loader)
        loss = train_step(model, x_0, mask_schedule, optimizer)
        pbar.set_postfix({"loss": f"{loss:.4f}"})
        if (step + 1) % sample_interval == 0:
            model.eval()
            with torch.no_grad():
                context_tokens = None
                if model.config.context_len > 0 and dataset_tokens is not None:
                    ctx_len = model.config.context_len
                    start_idx = torch.randint(0, len(dataset_tokens) - ctx_len, (1,)).item()
                    context_tokens = dataset_tokens[start_idx:start_idx + ctx_len].unsqueeze(0)
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
            checkpoint_path = f"weights/bpe_diffusion_model_step_{step + 1}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            model.train()

def main():
    batch_size = 64
    max_iters = 20000
    eval_interval = 1000
    learning_rate = 3e-4
    data_path = Path("data/tiny_shakespeare.txt")
    
    # 1. Initialize tokenizer FIRST
    print("Initializing tokenizer...")
    get_tokenizer(data_paths=[data_path])
    v_size = vocab_size()
    m_id = mask_token_id()
    print(f"Tokenizer ready. Vocab size: {v_size}, Mask ID: {m_id}\n")
    
    # 2. Create config
    config = DiffusionConfig(vocab_size=v_size, mask_token_id=m_id)
    print(f"Config: seq_len={config.sequence_len}, vocab={config.vocab_size}, mask_id={config.mask_token_id}")
    
    # 3. Device
    device = torch.device("cuda" if torch.cuda.is_available() else
                         ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}\n")
    
    # 4. Model
    model = DiffusionTransformer(config).to(device)
    model.init_weights()
    
    # 5. MPS warmup
    if device.type == "mps":
        print("Warming up MPS...")
        with torch.no_grad():
            dummy = torch.randint(0, config.vocab_size, (1, config.sequence_len), device=device)
            dummy_t = torch.tensor([0], device=device)
            _ = model(dummy, dummy_t)
        print("MPS ready!")
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,}\n")
    
    # 6. Training setup
    mask_schedule = MaskedDiffusionSchedule(config.diffusion_steps, config.mask_token_id, config.context_len)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # 7. Data loader
    print("Creating data loader...")
    data_loader = get_data_loader(str(data_path), batch_size, config.sequence_len, device)
    print("Data loader ready!\n")
    
    # 8. Context tokens
    dataset_tokens = None
    if config.context_len > 0:
        with open(data_path, "r", encoding="utf-8") as f:
            text = f.read()
        dataset_tokens = encode_text(text)
        print(f"Loaded {len(dataset_tokens)} tokens for context sampling\n")
    
    # 9. Train
    print("Starting training...\n")
    train(model, data_loader, mask_schedule, optimizer, max_iters, eval_interval, dataset_tokens)
    torch.save(model.state_dict(), "weights/diffusion_model.pt")
    print("Model saved to weights/diffusion_model.pt")

if __name__ == "__main__":
    main()
