"""
chat_tui.py - Simple diffusion chat with live unmasking animation
Supports both BPE tokenizer (-b) and ASCII character-level (-a)
Usage: 
  python chat_tui.py --model weights/diffusion_model.pt -b  # BPE mode
  python chat_tui.py --model weights/diffusion_model.pt -a  # ASCII mode
"""

import argparse
import sys
import torch
import torch.nn.functional as F
from model import DiffusionTransformer, DiffusionConfig
import time

USE_BPE = True

def load_model(checkpoint_path, device, use_bpe=True):
    """Load model from checkpoint with appropriate tokenizer"""
    global USE_BPE
    USE_BPE = use_bpe
    
    if use_bpe:
        from tokenizer_utils import get_tokenizer, vocab_size, mask_token_id
        get_tokenizer()
        v_size = vocab_size()
        m_id = mask_token_id()
    else:
        v_size = 128
        m_id = 0
    
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    try:
        config = DiffusionConfig(n_layer=5, n_head=4, n_embd=128, vocab_size=v_size, mask_token_id=m_id)
        model = DiffusionTransformer(config).to(device)
        model.load_state_dict(state_dict)
        model_type = "Student"
    except:
        config = DiffusionConfig(vocab_size=v_size, mask_token_id=m_id)
        model = DiffusionTransformer(config).to(device)
        model.load_state_dict(state_dict)
        model_type = "Teacher"
    
    model.eval()
    return model, config, model_type

def encode_text(text):
    """Encode text using current tokenizer mode"""
    if USE_BPE:
        from tokenizer_utils import encode_text as bpe_encode
        return bpe_encode(text)
    else:
        tokens = torch.tensor([min(ord(c), 127) for c in text], dtype=torch.long)
        return tokens

def decode_tokens(tokens):
    """Decode tokens using current tokenizer mode"""
    if USE_BPE:
        from tokenizer_utils import decode_tokens as bpe_decode
        return bpe_decode(tokens)
    else:
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        text = ''.join([chr(int(t)) for t in tokens])
        return text

def clear_screen():
    """Clear terminal screen"""
    print("\033[2J\033[H", end="")

def print_at_line(line_num, text):
    """Print text at specific line"""
    print(f"\033[{line_num};0H{text}\033[K", end="")
    sys.stdout.flush()

def generate_with_animation(model, context_text, device, temperature=1.0, confidence_threshold=0.9):
    """Generate text with proper real-time rendering"""
    seq_len = model.config.sequence_len
    mask_token_id = model.config.mask_token_id
    
    # Encode context
    context_tokens = None
    ctx_len = 0
    if context_text and model.config.context_len > 0:
        context_encoded = encode_text(context_text)
        context_len = min(len(context_encoded), model.config.context_len)
        context_tokens = context_encoded[-context_len:].unsqueeze(0).to(device)
        ctx_len = context_tokens.size(1)
    
    # Initialize
    x = torch.full((1, seq_len), mask_token_id, dtype=torch.long, device=device)
    if context_tokens is not None:
        x[:, :ctx_len] = context_tokens
    
    masked_positions = torch.ones(1, seq_len, dtype=torch.bool, device=device)
    if context_tokens is not None:
        masked_positions[:, :ctx_len] = False
    
    step = 0
    start_time = time.time()
    
    # Clear and setup display
    clear_screen()
    tokenizer_mode = "BPE Tokenizer" if USE_BPE else "ASCII Character-Level"
    print_at_line(1, "╔═══════════════════════════════════════════════════════════════╗")
    print_at_line(2, f"║          DIFFUSION GENERATION ({tokenizer_mode})              ║")
    print_at_line(3, "╚═══════════════════════════════════════════════════════════════╝")
    
    while masked_positions.any():
        t_batch = torch.full((1,), step, device=device, dtype=torch.long)
        t_batch = torch.clamp(t_batch, 0, model.config.diffusion_steps - 1)
        
        with torch.no_grad():
            logits = model.forward(x, t_batch)
            probs = F.softmax(logits / temperature, dim=-1)
            confidences, predicted_tokens = torch.max(probs, dim=-1)
            above_threshold = (confidences >= confidence_threshold) & masked_positions
            
            if not above_threshold.any():
                masked_confidences = confidences.clone()
                masked_confidences[~masked_positions] = -float("inf")
                best_idx = torch.argmax(masked_confidences[0])
                above_threshold[0, best_idx] = True
            
            x = torch.where(above_threshold, predicted_tokens, x)
            masked_positions = masked_positions & ~above_threshold
        
        # Create display by decoding entire sequence with masked tokens replaced
        # This preserves proper spacing and newlines
        display_tokens = x[0].clone()
        
        # Replace masked positions with a special placeholder
        # For BPE, we'll decode the unmasked parts and insert █ for masked
        if USE_BPE:
            # Decode only unmasked tokens, insert █ for masked
            # Build a hybrid representation
            text_parts = []
            current_segment = []
            
            for i in range(seq_len):
                if masked_positions[0, i]:
                    # Flush current segment
                    if current_segment:
                        segment_text = decode_tokens(torch.tensor(current_segment))
                        text_parts.append(segment_text)
                        current_segment = []
                    text_parts.append("█")
                else:
                    current_segment.append(display_tokens[i].item())
            
            # Flush remaining
            if current_segment:
                segment_text = decode_tokens(torch.tensor(current_segment))
                text_parts.append(segment_text)
            
            text = "".join(text_parts)
        else:
            # ASCII mode - decode character by character
            display_parts = []
            for i in range(seq_len):
                if masked_positions[0, i]:
                    display_parts.append("█")
                else:
                    char = chr(int(display_tokens[i].item()))
                    display_parts.append(char)
            text = "".join(display_parts)
        
        # Format text for display - preserve newlines and spaces
        # Replace actual newlines with visible markers for display
        display_text = text.replace("\n", "↵\n")
        
        # Split into lines (respecting actual newlines)
        lines = display_text.split("\n")
        
        # Wrap long lines at 60 chars
        wrapped_lines = []
        for line in lines:
            if len(line) <= 60:
                wrapped_lines.append(line)
            else:
                # Wrap at 60 chars
                for i in range(0, len(line), 60):
                    wrapped_lines.append(line[i:i+60])
        
        # Display (max 15 lines)
        for idx, line in enumerate(wrapped_lines[:15]):
            print_at_line(5 + idx, f"  {line}")
        
        # Clear remaining lines
        for idx in range(len(wrapped_lines[:15]), 15):
            print_at_line(5 + idx, "  " + " "*60)
        
        # Stats
        num_masked = masked_positions.sum().item()
        elapsed = time.time() - start_time
        print_at_line(22, f"  Step: {step:3d} | Masked: {num_masked:3d}/{seq_len} | Time: {elapsed:.1f}s")
        
        step += 1
        time.sleep(0.05)  # Slightly slower for readability
    
    # Final text
    final_text = decode_tokens(x[0])
    
    # Truncate at EOS or reasonable length
    if "[EOS]" in final_text:
        final_text = final_text.split("[EOS]")[0]
    elif len(final_text) > 500:
        for delim in [". ", "! ", "? ", "\n\n"]:
            idx = final_text[:500].rfind(delim)
            if idx > 0:
                final_text = final_text[:idx+1]
                break
    
    return final_text

def main():
    parser = argparse.ArgumentParser(description="Diffusion chat with BPE or ASCII tokenizer")
    parser.add_argument("--model", type=str, default="weights/diffusion_model.pt")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--confidence", type=float, default=0.9)
    
    tokenizer_group = parser.add_mutually_exclusive_group()
    tokenizer_group.add_argument("-b", "--bpe", action="store_true")
    tokenizer_group.add_argument("-a", "--ascii", action="store_true")
    
    args = parser.parse_args()
    
    use_bpe = not args.ascii
    
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         ("mps" if torch.backends.mps.is_available() else "cpu"))
    
    tokenizer_name = "BPE" if use_bpe else "ASCII"
    print(f"Loading model from {args.model} with {tokenizer_name} tokenizer...")
    model, config, model_type = load_model(args.model, device, use_bpe)
    print(f"Model: {model_type} | Vocab: {config.vocab_size} | Seq: {config.sequence_len}")
    print(f"Tokenizer: {tokenizer_name}")
    
    conversation = ""
    
    while True:
        print("\n" + "="*60)
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        
        if user_input.lower() == "clear":
            conversation = ""
            print("Conversation cleared!")
            continue
        
        conversation += f"Human: {user_input}\nAI: "
        
        response = generate_with_animation(
            model, conversation, device,
            args.temperature, args.confidence
        )
        
        if "AI:" in response:
            ai_response = response.split("AI:")[-1].strip()
        else:
            ai_response = response.strip()
        
        conversation += ai_response + "\n"
        
        clear_screen()
        print("\n" + "="*60)
        print(f"AI: {ai_response}")
        print("="*60)

if __name__ == "__main__":
    main()
