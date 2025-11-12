"""
Preprocess conversational data to add [TURN] boundary tokens.
Standardizes speaker labels and adds explicit turn markers.
"""

import re
from pathlib import Path

def preprocess_conversations(input_file: Path, output_file: Path):
    """Add [TURN] tokens and standardize speaker labels"""
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Original text length: {len(text)} characters")
    
    # Standardize speaker labels: Human 1 -> Human, Human 2 -> AI
    text = re.sub(r'Human\s*1\s*:', '[TURN]Human:', text)
    text = re.sub(r'Human\s*2\s*:', '[TURN]AI:', text)
    
    # Also handle cases where there's no space
    text = text.replace('Human1:', '[TURN]Human:')
    text = text.replace('Human2:', '[TURN]AI:')
    
    # Remove any double [TURN] tokens that might have been created
    text = text.replace('[TURN][TURN]', '[TURN]')
    
    # Ensure we start with [TURN]
    if not text.startswith('[TURN]'):
        text = '[TURN]' + text
    
    print(f"Processed text length: {len(text)} characters")
    print(f"Number of turns: {text.count('[TURN]')}")
    
    # Save processed text
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(text)
    
    print(f"Saved to {output_file}")
    
    # Show sample
    print("\nSample of processed text:")
    print(text[:500])

if __name__ == "__main__":
    input_path = Path("data/tiny_shakespeare.txt")
    output_path = Path("data/conversations_with_turns.txt")
    
    preprocess_conversations(input_path, output_path)
