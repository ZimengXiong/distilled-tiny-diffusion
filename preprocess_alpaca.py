"""
Preprocess alpaca Q&A data for 128-token training
Filters out QA pairs that are too long
"""

from pathlib import Path
import re
from tokenizer_utils import get_tokenizer, encode_text

def preprocess_alpaca_data(input_file, output_file, max_tokens=120):
    """Filter and format alpaca QA data"""
    
    # Initialize tokenizer
    print("Initializing tokenizer...")
    get_tokenizer(data_paths=[input_file])
    
    # Read data
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Split into QA pairs (split on double newlines or when you see Human 1:)
    qa_pairs = re.split(r'\n\s*\n+(?=Human 1:)', text.strip())
    qa_pairs = [p.strip() for p in qa_pairs if p.strip()]
    
    print(f"Found {len(qa_pairs)} QA pairs")
    
    # Filter by token length
    filtered = []
    for pair in qa_pairs:
        # Encode to check length
        tokens = encode_text(pair)
        
        if len(tokens) <= max_tokens:
            filtered.append(pair)
        else:
            print(f"Skipping long pair ({len(tokens)} tokens): {pair[:50]}...")
    
    print(f"\nKept {len(filtered)}/{len(qa_pairs)} pairs (≤{max_tokens} tokens)")
    
    # Join with spaces for training
    output_text = ' '.join(filtered)
    
    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(output_text)
    
    print(f"Saved to {output_file}")
    print(f"\nSample output:")
    print(output_text[:300])

if __name__ == "__main__":
    input_path = Path("data/alpaca_facts.txt")
    output_path = Path("data/alpaca_processed.txt")
    
    preprocess_alpaca_data(input_path, output_path, max_tokens=120)
