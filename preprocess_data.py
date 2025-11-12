"""
Preprocess data to single-turn QA format
Only keeps conversations where both turns fit in 128 tokens
"""

from pathlib import Path
import re
from tokenizer_utils import get_tokenizer, encode_text

def preprocess_single_turn(input_file, output_file, max_tokens=128):
    """Convert to single-turn QA, filter by length"""
    
    # Initialize tokenizer
    get_tokenizer(data_paths=[input_file])
    
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Split into conversation turns
    turns = re.split(r'(Human [12]\s*:)', text)
    
    # Group into pairs
    pairs = []
    i = 0
    while i < len(turns) - 2:
        if 'Human 1' in turns[i]:
            speaker1 = turns[i]
            content1 = turns[i+1].strip()
            
            if i+2 < len(turns) and 'Human 2' in turns[i+2]:
                speaker2 = turns[i+2]
                content2 = turns[i+3].strip() if i+3 < len(turns) else ""
                
                # Create pair
                input_text = f"Human 1: {content1}"
                output_text = f"Human 2: {content2}"
                
                # Check token length
                input_tokens = encode_text(input_text)
                output_tokens = encode_text(output_text)
                
                total_len = len(input_tokens) + len(output_tokens)
                
                # Only keep if fits in 128 tokens
                if total_len <= max_tokens and len(content2) > 5:
                    pairs.append(f"{input_text} {output_text}")
                
                i += 4
            else:
                i += 2
        else:
            i += 1
    
    print(f"Original pairs: ~{len(turns)//4}")
    print(f"Filtered pairs (≤{max_tokens} tokens): {len(pairs)}")
    
    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(' '.join(pairs))
    
    print(f"\nSaved to {output_file}")
    print(f"Sample: {pairs[0][:200]}")

if __name__ == "__main__":
    input_path = Path("data/tiny_shakespeare.txt")
    output_path = Path("data/conversations_128.txt")
    
    preprocess_single_turn(input_path, output_path, max_tokens=128)
