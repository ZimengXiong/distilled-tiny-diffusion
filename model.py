"""
Simple BPE-Tokenized Discrete Diffusion Transformer
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizer_utils import (
    decode_tokens as bpe_decode_tokens,
    encode_text as bpe_encode_text,
)

@dataclass
class DiffusionConfig:
    sequence_len: int = 128
    vocab_size: int = 4096  # Default, override in training
    mask_token_id: int = 1   # Default, override in training
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    diffusion_steps: int = 128
    context_len: int = 64

def norm(x):
    return F.rms_norm(x, (x.size(-1),))

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], 3)
    out = out.to(x.dtype)
    return out

class BidirectionalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = BidirectionalAttention(config)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin):
        x = x + self.attn(norm(x), cos_sin)
        x = x + self.mlp(norm(x))
        return x

class DiffusionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.time_emb = nn.Embedding(config.diffusion_steps, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.output_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.rotary_seq_len = config.sequence_len * 2
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def init_weights(self):
        self.apply(self._init_weights)
        torch.nn.init.zeros_(self.output_head.weight)
        for block in self.blocks:
            if isinstance(block, Block):
                torch.nn.init.zeros_(block.mlp.c_proj.weight)
                torch.nn.init.zeros_(block.attn.c_proj.weight)
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.token_emb.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def get_device(self):
        return self.token_emb.weight.device

    def forward(self, x_t, t):
        B, T = x_t.size()
        x = self.token_emb(x_t)
        t_emb = self.time_emb(t)
        x = x + t_emb.unsqueeze(1)
        x = norm(x)
        assert T <= self.cos.size(1)
        cos_sin = (self.cos[:, :T], self.sin[:, :T])
        for block in self.blocks:
            x = block(x, cos_sin)
        x = norm(x)
        logits = self.output_head(x)
        return logits

    @torch.inference_mode()
    def sample_topk(self, batch_size, seq_len, k, num_steps=None, temperature=1.0, device=None, context_tokens=None):
        if device is None:
            device = self.get_device()
        if num_steps is None:
            num_steps = seq_len
        x = torch.full((batch_size, seq_len), self.config.mask_token_id, dtype=torch.long, device=device)
        context_len = 0
        if context_tokens is not None:
            context_tokens = context_tokens.to(device)
            context_len = context_tokens.size(1)
            x[:, :context_len] = context_tokens
        masked_positions = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
        if context_len > 0:
            masked_positions[:, :context_len] = False
        for step in range(num_steps):
            if not masked_positions.any():
                break
            t_batch = torch.full((batch_size,), step, device=device, dtype=torch.long)
            t_batch = torch.clamp(t_batch, 0, self.config.diffusion_steps - 1)
            logits = self.forward(x, t_batch)
            probs = F.softmax(logits / temperature, dim=-1)
            confidences, predicted_tokens = torch.max(probs, dim=-1)
            confidences = confidences.masked_fill(~masked_positions, -float("inf"))
            remaining = masked_positions.sum(dim=1).max().item()
            k_actual = int(min(k, max(1, int(remaining))))
            _, topk_indices = torch.topk(confidences, k=k_actual, dim=1)
            for b in range(batch_size):
                for idx in topk_indices[b]:
                    if masked_positions[b, idx]:
                        x[b, idx] = predicted_tokens[b, idx]
                        masked_positions[b, idx] = False
        return x

    @torch.inference_mode()
    def sample_confidence(self, batch_size, seq_len, confidence_threshold=0.95, num_steps=None, temperature=1.0, device=None, context_tokens=None):
        if device is None:
            device = self.get_device()
        if num_steps is None:
            num_steps = seq_len
        x = torch.full((batch_size, seq_len), self.config.mask_token_id, dtype=torch.long, device=device)
        context_len = 0
        if context_tokens is not None:
            context_tokens = context_tokens.to(device)
            context_len = context_tokens.size(1)
            x[:, :context_len] = context_tokens
        masked_positions = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
        if context_len > 0:
            masked_positions[:, :context_len] = False
        for step in range(num_steps):
            if not masked_positions.any():
                break
            t_batch = torch.full((batch_size,), step, device=device, dtype=torch.long)
            t_batch = torch.clamp(t_batch, 0, self.config.diffusion_steps - 1)
            logits = self.forward(x, t_batch)
            probs = F.softmax(logits / temperature, dim=-1)
            confidences, predicted_tokens = torch.max(probs, dim=-1)
            above_threshold = (confidences >= confidence_threshold) & masked_positions
            for b in range(batch_size):
                if masked_positions[b].any() and not above_threshold[b].any():
                    masked_confidences = confidences[b].clone()
                    masked_confidences[~masked_positions[b]] = -float("inf")
                    best_idx = torch.argmax(masked_confidences)
                    above_threshold[b, best_idx] = True
            x = torch.where(above_threshold, predicted_tokens, x)
            masked_positions = masked_positions & ~above_threshold
        return x

    @torch.inference_mode()
    def sample(self, batch_size, seq_len, num_steps=None, temperature=1.0, device=None, context_tokens=None, method="confidence", k=None, confidence_threshold=0.95):
        if method == "topk":
            if k is None:
                k = max(1, seq_len // 10)
            return self.sample_topk(batch_size, seq_len, k, num_steps, temperature, device, context_tokens)
        elif method == "confidence":
            return self.sample_confidence(batch_size, seq_len, confidence_threshold, num_steps, temperature, device, context_tokens)
        else:
            raise ValueError(f"Unknown sampling method: {method}")

def encode_text(text, *, add_bos=False, add_eos=False):
    return bpe_encode_text(text, add_bos=add_bos, add_eos=add_eos)

def decode_tokens(tokens):
    return bpe_decode_tokens(tokens)
