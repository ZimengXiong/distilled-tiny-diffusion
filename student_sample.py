import torch
from model import DiffusionTransformer, DiffusionConfig, decode_tokens, encode_text

def load_dataset_text(data_path="data/tiny_shakespeare.txt"):
    """Load dataset text for random context sampling"""
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()
    return encode_text(text)

def get_random_context(dataset_tokens, context_len, batch_size=1):
    """Get random context tokens from dataset"""
    max_start = len(dataset_tokens) - context_len
    start_indices = torch.randint(0, max_start, (batch_size,))
    context_tokens = torch.stack([
        dataset_tokens[start : start + context_len] for start in start_indices
    ])
    return context_tokens

def load_student_model(checkpoint_path, device):
    """Load distilled student model from checkpoint"""
    # Use distilled config: match distillation!
    student_config = DiffusionConfig(n_layer=5, n_head=4, n_embd=128)
    model = DiffusionTransformer(student_config).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

def generate_samples(
    model,
    num_samples=5,
    temperature=1.0,
    dataset_tokens=None,
    method="confidence",
    k=None,
    confidence_threshold=0.95,
):
    """Generate text samples from the student model"""
    device = next(model.parameters()).device
    method_desc = f"{method}"
    if method == "topk":
        if k is None:
            k = max(1, model.config.sequence_len // 10)
        method_desc += f" (K={k})"
    elif method == "confidence":
        method_desc += f" (τ={confidence_threshold})"
    print(f"Generating {num_samples} samples using {method_desc} decoding...\n")
    for i in range(num_samples):
        with torch.no_grad():
            context_tokens = None
            if model.config.context_len > 0 and dataset_tokens is not None:
                context_tokens = get_random_context(
                    dataset_tokens, model.config.context_len, batch_size=1
                )
            tokens = model.sample(
                batch_size=1,
                seq_len=model.config.sequence_len,
                num_steps=None,
                temperature=temperature,
                device=device,
                context_tokens=context_tokens,
                method=method,
                k=k,
                confidence_threshold=confidence_threshold,
            )
            text = decode_tokens(tokens[0])
            print(f"--- Sample {i + 1} ---")
            print(text)
            print()

def generate_continuous_blocks(
    model,
    num_blocks=30,
    temperature=1.0,
    dataset_tokens=None,
    method="confidence",
    k=None,
    confidence_threshold=0.95,
):
    """
    Generate multiple blocks sequentially, conditioned on last context_len chars of previous block.
    """
    device = next(model.parameters()).device
    context_len = model.config.context_len
    method_desc = f"{method}"
    if method == "topk":
        if k is None:
            k = max(1, model.config.sequence_len // 10)
        method_desc += f" (K={k})"
    elif method == "confidence":
        method_desc += f" (τ={confidence_threshold})"
    print(f"Generating {num_blocks} continuous blocks using {method_desc} decoding...")
    print(f"Each block conditions on the last {context_len} characters of previous block\n")
    all_text = ""
    prev_context = None
    for block_idx in range(num_blocks):
        with torch.no_grad():
            context_tokens = None
            if block_idx == 0 and dataset_tokens is not None:
                context_tokens = get_random_context(
                    dataset_tokens, context_len, batch_size=1
                )
            elif block_idx > 0:
                context_tokens = prev_context.unsqueeze(0)
            tokens = model.sample(
                batch_size=1,
                seq_len=model.config.sequence_len,
                num_steps=None,
                temperature=temperature,
                device=device,
                context_tokens=context_tokens,
                method=method,
                k=k,
                confidence_threshold=confidence_threshold,
            )
            prev_context = tokens[0, -context_len:]
            text = decode_tokens(tokens[0])
            if block_idx == 0:
                print(text, end="", flush=True)
                all_text += text
            else:
                new_text = text[context_len:]
                print(new_text, end="", flush=True)
                all_text += new_text
    print("\n")
    return all_text

def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}\n")
    checkpoint_path = "weights/student_1M.pt"  # student weights
    print(f"Loading student model from {checkpoint_path}...")
    model = load_student_model(checkpoint_path, device)
    print("Student model loaded!\n")
    dataset_tokens = None
    if model.config.context_len > 0:
        print("Loading dataset for context sampling...")
        dataset_tokens = load_dataset_text("data/tiny_shakespeare.txt")
        print(f"Loaded {len(dataset_tokens)} tokens from dataset\n")
    method = "confidence"
    confidence_threshold = 0.9
    k = 1      # for topk method (otherwise None)
    if model.config.context_len > 0:
        print(f"Using continuous block generation (context_len={model.config.context_len})\n")
        generate_continuous_blocks(
            model,
            num_blocks=30,
            temperature=1.0,
            dataset_tokens=dataset_tokens,
            method=method,
            k=k,
            confidence_threshold=confidence_threshold,
        )
    else:
        print("Using independent sample generation (no context)\n")
        generate_samples(
            model,
            num_samples=5,
            temperature=1.0,
            dataset_tokens=dataset_tokens,
            method=method,
            k=k,
            confidence_threshold=confidence_threshold,
        )

if __name__ == "__main__":
    main()
