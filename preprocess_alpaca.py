"""
Preprocess alpaca Q&A data with [EOS] and [PAD] tokens
Each sample is exactly 128 tokens: "Human 1: Q Human 2: A [EOS] [PAD]..."
"""

from pathlib import Path
import re
from tokenizer_utils import get_tokenizer, encode_text, decode_tokens

def preprocess_with_eos_and_padding(input_file, output_file, max_tokens=128):
    """Add [EOS] after each answer, then pad to 128 tokens"""
    
    print("Initializing tokenizer...")
    tokenizer = get_tokenizer(data_paths=[input_file])
    
    pad_token_id = tokenizer.token_to_id("[PAD]")
    eos_token_id = tokenizer.token_to_id("[EOS]")
    
    print(f"PAD token ID: {pad_token_id}")
    print(f"EOS token ID: {eos_token_id}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Split into QA pairs (double newlines separate them)
    qa_pairs = re.split(r'\n\s*\n+(?=Human 1:)', text.strip())
    qa_pairs = [p.strip() for p in qa_pairs if p.strip()]
    
    print(f"Found {len(qa_pairs)} QA pairs")
    
    padded_samples = []
    skipped = 0
    
    for pair in qa_pairs:
        # Add [EOS] token after the answer
        pair_with_eos = pair + " [EOS]"
        
        # Encode
        tokens = encode_text(pair_with_eos)
        
        if len(tokens) > max_tokens:
            print(f"Skipping long pair ({len(tokens)} tokens): {pair[:50]}...")
            skipped += 1
            continue
        
        # Pad to exactly max_tokens
        padding_needed = max_tokens - len(tokens)
        padded_tokens = tokens.tolist() + [pad_token_id] * padding_needed
        
        # Store as space-separated token IDs
        padded_samples.append(' '.join(map(str, padded_tokens)))
    
    print(f"\nKept {len(padded_samples)}/{len(qa_pairs)} pairs")
    print(f"Skipped {skipped} pairs that were too long")
    
    # Save - each line is one 128-token sample
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(padded_samples))
    
    print(f"Saved to {output_file}")
    
    # Verify first sample
    first_sample = [int(t) for t in padded_samples[0].split()]
    decoded = decode_tokens(first_sample)
    print(f"\nFirst sample ({len(first_sample)} tokens):")
    print(decoded[:200])

if __name__ == "__main__":
    input_path = Path("data/alpaca_facts.txt")
    output_path = Path("data/alpaca_padded.txt")
    
    preprocess_with_eos_and_padding(input_path, output_path, max_tokens=128)
