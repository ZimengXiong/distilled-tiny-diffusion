"""
chat.py - Alpaca expert chat with [EOS] stopping and animation
"""

import argparse
import torch
import torch.nn.functional as F
from model import DiffusionTransformer, DiffusionConfig
import time
import re

torch.serialization.add_safe_globals([DiffusionConfig])

def load_model(checkpoint_path, device):
    from tokenizer_utils import get_tokenizer, vocab_size, mask_token_id
    get_tokenizer()
    v_size, m_id = vocab_size(), mask_token_id()
    
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_state = state_dict.get('model_state_dict', state_dict)
    config_saved = state_dict.get('config', None)
    
    if config_saved:
        config = config_saved
        config.vocab_size, config.mask_token_id = v_size, m_id
    else:
        config = DiffusionConfig(vocab_size=v_size, mask_token_id=m_id)
    
    model = DiffusionTransformer(config).to(device)
    model.load_state_dict(model_state)
    model.eval()
    return model, config

def encode_text(text):
    from tokenizer_utils import encode_text as bpe_encode
    return bpe_encode(text)

def decode_tokens(tokens):
    from tokenizer_utils import decode_tokens as bpe_decode
    return bpe_decode(tokens)

def extract_first_response(full_text):
    """Extract first AI response, stop at [EOS], remove [PAD]"""
    
    # Remove [PAD] tokens
    full_text = re.sub(r'\[PAD\]', '', full_text)
    
    # Stop at [EOS]
    if "[EOS]" in full_text:
        full_text = full_text.split("[EOS]")[0]
    
    # Find first Human 2:
    if "Human 2:" not in full_text:
        clean = re.sub(r'Human [12]\s*:', '', full_text).strip()
        return clean[:100] if clean else "..."
    
    # Get text after first Human 2:
    parts = full_text.split("Human 2:")
    ai_text = parts[1] if len(parts) > 1 else parts[0]
    
    # Stop at next speaker marker
    for marker in ["Human 1:", "Human 2:"]:
        if marker in ai_text:
            ai_text = ai_text.split(marker)[0]
            break
    
    # Clean up
    ai_text = ai_text.strip()
    ai_text = re.sub(r'Human [12]\s*:', '', ai_text).strip()
    
    return ai_text if ai_text else "..."

def clear_screen():
    print("\033[2J\033[H", end="", flush=True)

def print_at(row, col, text):
    print(f"\033[{row};{col}H{text}", end="", flush=True)

def generate_response(model, user_input, device, temperature=1.0, confidence=0.9):
    """Generate with [EOS] early stopping and animation"""
    
    from tokenizer_utils import get_tokenizer
    tokenizer = get_tokenizer()
    eos_token_id = tokenizer.token_to_id("[EOS]")
    pad_token_id = tokenizer.token_to_id("[PAD]")
    
    # Format input
    input_text = f"Human 1: {user_input} Human 2: "
    input_tokens = encode_text(input_text)
    
    # Truncate if needed
    max_context = 64
    if len(input_tokens) > max_context:
        input_tokens = input_tokens[-max_context:]
    
    context_tokens = input_tokens.unsqueeze(0).to(device)
    ctx_len = context_tokens.size(1)
    
    # Initialize
    seq_len = 128
    mask_token_id = model.config.mask_token_id
    
    x = torch.full((1, seq_len), mask_token_id, dtype=torch.long, device=device)
    x[:, :ctx_len] = context_tokens
    
    masked_positions = torch.ones(1, seq_len, dtype=torch.bool, device=device)
    masked_positions[:, :ctx_len] = False
    
    step = 0
    start_time = time.time()
    eos_found = False
    
    # Setup display
    clear_screen()
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║                    DIFFUSION GENERATION                            ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print()
    
    # Generation loop with animation
    while masked_positions.any() and not eos_found:
        t_batch = torch.full((1,), step, device=device, dtype=torch.long)
        t_batch = torch.clamp(t_batch, 0, model.config.diffusion_steps - 1)
        
        with torch.no_grad():
            logits = model.forward(x, t_batch)
            probs = F.softmax(logits / temperature, dim=-1)
            confidences, predicted_tokens = torch.max(probs, dim=-1)
            above_threshold = (confidences >= confidence) & masked_positions
            
            if not above_threshold.any():
                masked_confidences = confidences.clone()
                masked_confidences[~masked_positions] = -float("inf")
                best_idx = torch.argmax(masked_confidences[0])
                above_threshold[0, best_idx] = True
            
            x = torch.where(above_threshold, predicted_tokens, x)
            masked_positions = masked_positions & ~above_threshold
            
            # Check for [EOS]
            if (x[0, ctx_len:] == eos_token_id).any():
                eos_found = True
        
        # Animation every 2 steps
        if step % 2 == 0:
            # Decode current state
            text_parts = []
            current_segment = []
            
            for i in range(seq_len):
                if masked_positions[0, i]:
                    if current_segment:
                        text_parts.append(decode_tokens(torch.tensor(current_segment)))
                        current_segment = []
                    text_parts.append("█")
                else:
                    current_segment.append(x[0, i].item())
            
            if current_segment:
                text_parts.append(decode_tokens(torch.tensor(current_segment)))
            
            current_text = "".join(text_parts)
            
            # Extract AI response only
            display = extract_first_response(current_text)
            
            # Wrap at 66 chars
            lines = []
            for i in range(0, len(display), 66):
                lines.append(display[i:i+66])
            
            # Display up to 12 lines
            for idx in range(12):
                if idx < len(lines):
                    print_at(5 + idx, 1, f"  {lines[idx]:<66}")
                else:
                    print_at(5 + idx, 1, " " * 70)
            
            # Stats
            num_masked = masked_positions.sum().item()
            elapsed = time.time() - start_time
            print_at(19, 1, "─" * 70)
            
            status = "[EOS FOUND - STOPPING]" if eos_found else ""
            print_at(20, 1, f"  Step: {step:3d} | Masked: {num_masked:3d}/{seq_len} | Time: {elapsed:.1f}s {status:<20}" + " " * 5)
        
        step += 1
        time.sleep(0.015)
    
    # Final decode
    full_text = decode_tokens(x[0])
    ai_response = extract_first_response(full_text)
    
    # Move cursor below animation
    print_at(22, 1, "")
    print("\n" + "="*70)
    
    return ai_response

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--confidence", type=float, default=0.9)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         ("mps" if torch.backends.mps.is_available() else "cpu"))
    
    print(f"Loading model...")
    model, config = load_model(args.model, device)
    print(f"Ready! Seq: {config.sequence_len} tokens\n")
    
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║                    ALPACA EXPERT CHAT                             ║")
    print("║  Type 'quit' to exit                                              ║")
    print("╚════════════════════════════════════════════════════════════════════╝\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() in ["quit", "exit", "q"]:
            print("\nGoodbye!")
            break
        
        ai_response = generate_response(model, user_input, device, args.temperature, args.confidence)
        
        print(f"\nAI: {ai_response}")
        print("="*70 + "\n")

if __name__ == "__main__":
    main()
