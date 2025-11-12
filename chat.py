"""
chat_tui.py - Interactive TUI to chat with diffusion model
Shows live denoising animation during generation
Usage: python chat_tui.py --model weights/diffusion_model.pt
"""

import argparse
import torch
import torch.nn.functional as F
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
from rich.prompt import Prompt
from rich.syntax import Syntax
from model import DiffusionTransformer, DiffusionConfig, encode_text, decode_tokens
import time

console = Console()

def load_model(checkpoint_path, device):
    """Load model from checkpoint"""
    # Try student config first, fall back to teacher
    try:
        config = DiffusionConfig(n_layer=5, n_head=4, n_embd=128)  # Student
        model = DiffusionTransformer(config).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model_type = "Student (1M params)"
    except:
        config = DiffusionConfig()  # Teacher
        model = DiffusionTransformer(config).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model_type = "Teacher (10.7M params)"
    
    model.eval()
    return model, config, model_type

def generate_with_animation(model, context_text, console, temperature=1.0, confidence_threshold=0.9):
    """Generate text with live diffusion animation"""
    device = next(model.parameters()).device
    seq_len = model.config.sequence_len
    mask_token_id = model.config.mask_token_id
    
    # Encode context
    context_tokens = None
    if context_text and model.config.context_len > 0:
        context_encoded = encode_text(context_text)
        context_len = min(len(context_encoded), model.config.context_len)
        context_tokens = context_encoded[-context_len:].unsqueeze(0).to(device)
    
    # Initialize masked sequence
    x = torch.full((1, seq_len), mask_token_id, dtype=torch.long, device=device)
    if context_tokens is not None:
        ctx_len = context_tokens.size(1)
        x[:, :ctx_len] = context_tokens
    
    masked_positions = torch.ones(1, seq_len, dtype=torch.bool, device=device)
    if context_tokens is not None:
        masked_positions[:, :ctx_len] = False
    
    # Create live display
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="output", size=12),
        Layout(name="stats", size=3)
    )
    
    step = 0
    start_time = time.time()
    
    with Live(layout, console=console, refresh_per_second=10) as live:
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
            
            # Update display
            current_text = ""
            for i in range(seq_len):
                if x[0, i] == mask_token_id:
                    current_text += "█"
                else:
                    char = decode_tokens([x[0, i].item()])
                    current_text += char if char != "\n" else "↵"
            
            # Wrap text at 80 chars
            wrapped = []
            for i in range(0, len(current_text), 80):
                wrapped.append(current_text[i:i+80])
            
            num_masked = masked_positions.sum().item()
            elapsed = time.time() - start_time
            
            layout["header"].update(
                Panel(f"[bold cyan]Diffusion Step {step}[/bold cyan]", border_style="cyan")
            )
            layout["output"].update(
                Panel("\n".join(wrapped), title="Generation Progress", border_style="green")
            )
            layout["stats"].update(
                Panel(
                    f"Masked: {num_masked}/{seq_len} | Elapsed: {elapsed:.1f}s | Step: {step}",
                    border_style="yellow"
                )
            )
            
            step += 1
    
    final_text = decode_tokens(x[0])
    return final_text

def main():
    parser = argparse.ArgumentParser(description="Interactive TUI chat with diffusion model")
    parser.add_argument("--model", type=str, default="weights/diffusion_model.pt",
                       help="Path to model checkpoint (.pt file)")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Sampling temperature")
    parser.add_argument("--confidence", type=float, default=0.9,
                       help="Confidence threshold for decoding")
    args = parser.parse_args()
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))
    
    # Load model
    console.print(f"\n[bold green]Loading model from {args.model}...[/bold green]")
    model, config, model_type = load_model(args.model, device)
    
    # Welcome screen
    console.clear()
    console.print(Panel.fit(
        f"[bold cyan]Diffusion Model Chat Interface[/bold cyan]\n\n"
        f"Model: {model_type}\n"
        f"Device: {device}\n"
        f"Sequence Length: {config.sequence_len}\n"
        f"Context Length: {config.context_len}\n\n"
        f"[dim]Type your message and watch the diffusion process!\n"
        f"Commands: 'quit' to exit, 'clear' to reset[/dim]",
        border_style="cyan"
    ))
    
    conversation_history = ""
    
    while True:
        # Get user input
        user_input = Prompt.ask("\n[bold blue]You[/bold blue]")
        
        if user_input.lower() == "quit":
            console.print("[yellow]Goodbye![/yellow]")
            break
        
        if user_input.lower() == "clear":
            conversation_history = ""
            console.clear()
            console.print("[green]Conversation cleared![/green]")
            continue
        
        # Update conversation
        conversation_history += f"Human: {user_input}\nAI: "
        
        # Generate response with animation
        console.print("\n[bold green]AI is thinking...[/bold green]\n")
        
        response = generate_with_animation(
            model, 
            conversation_history,
            console,
            temperature=args.temperature,
            confidence_threshold=args.confidence
        )
        
        # Extract just the AI's response
        if "AI:" in response:
            ai_response = response.split("AI:")[-1].strip()
        else:
            ai_response = response.strip()
        
        # Display final response
        console.print(Panel(
            ai_response[:500],  # Limit display length
            title="[bold green]AI Response[/bold green]",
            border_style="green"
        ))
        
        conversation_history += ai_response + "\n"

if __name__ == "__main__":
    main()
