import time
import torch
from model import DiffusionTransformer, DiffusionConfig, decode_tokens, encode_text

def load_models(teacher_path, student_path, device):
    teacher_config = DiffusionConfig()
    teacher = DiffusionTransformer(teacher_config).to(device)
    teacher.load_state_dict(torch.load(teacher_path, map_location=device))
    teacher.eval()
    
    student_config = DiffusionConfig(n_layer=5, n_head=4, n_embd=128)
    student = DiffusionTransformer(student_config).to(device)
    student.load_state_dict(torch.load(student_path, map_location=device))
    student.eval()
    
    return teacher, student

def benchmark_model(model, context_tokens, num_runs=5, temperature=1.0, 
                   method="confidence", confidence_threshold=0.95):
    device = next(model.parameters()).device
    seq_len = model.config.sequence_len
    
    times = []
    tokens_generated = []
    
    print(f"  Running {num_runs} iterations...")
    
    for run in range(num_runs):
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        with torch.no_grad():
            start_time = time.perf_counter()
            
            tokens = model.sample(
                batch_size=1,
                seq_len=seq_len,
                temperature=temperature,
                device=device,
                context_tokens=context_tokens.to(device) if context_tokens is not None else None,
                method=method,
                confidence_threshold=confidence_threshold,
            )
            
            end_time = time.perf_counter()
        
        elapsed = end_time - start_time
        times.append(elapsed)
        tokens_generated.append(seq_len)
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    avg_tokens = sum(tokens_generated) / len(tokens_generated)
    tokens_per_sec = avg_tokens / avg_time
    
    return {
        'avg_time': avg_time,
        'min_time': min_time,
        'max_time': max_time,
        'tokens_per_sec': tokens_per_sec,
        'total_tokens': avg_tokens,
        'sample_output': decode_tokens(tokens[0])
    }

def compare_models(teacher_path="weights/diffusion_model.pt",
                  student_path="weights/student_1M.pt",
                  data_path="data/tiny_shakespeare.txt",
                  num_runs=5):
    """Compare teacher and student models on same input"""
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"\n{'='*70}")
    print(f"MODEL COMPARISON BENCHMARK")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Runs per model: {num_runs}\n")
    
    print("Loading models...")
    teacher, student = load_models(teacher_path, student_path, device)
    
    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())
    
    print(f"Teacher parameters: {teacher_params:,}")
    print(f"Student parameters: {student_params:,}")
    print(f"Compression ratio: {teacher_params / student_params:.2f}x\n")
    
    context_tokens = None
    if teacher.config.context_len > 0:
        with open(data_path, "r", encoding="utf-8") as f:
            text = f.read()
        dataset_tokens = encode_text(text)
        context_len = teacher.config.context_len
        start_idx = torch.randint(0, len(dataset_tokens) - context_len, (1,)).item()
        context_tokens = dataset_tokens[start_idx:start_idx + context_len].unsqueeze(0)
        print(f"Using context: {decode_tokens(context_tokens[0])[:100]}...\n")
    
    print("Benchmarking TEACHER model...")
    teacher_metrics = benchmark_model(teacher, context_tokens, num_runs)
    
    print("Benchmarking STUDENT model...")
    student_metrics = benchmark_model(student, context_tokens, num_runs)
    
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}\n")
    
    print(f"{'Metric':<25} {'Teacher':<20} {'Student':<20} {'Speedup':<10}")
    print(f"{'-'*70}")
    print(f"{'Average Time (s)':<25} {teacher_metrics['avg_time']:<20.4f} {student_metrics['avg_time']:<20.4f} {teacher_metrics['avg_time']/student_metrics['avg_time']:<10.2f}x")
    print(f"{'Min Time (s)':<25} {teacher_metrics['min_time']:<20.4f} {student_metrics['min_time']:<20.4f} {teacher_metrics['min_time']/student_metrics['min_time']:<10.2f}x")
    print(f"{'Max Time (s)':<25} {teacher_metrics['max_time']:<20.4f} {student_metrics['max_time']:<20.4f} {teacher_metrics['max_time']/student_metrics['max_time']:<10.2f}x")
    print(f"{'Tokens/Second':<25} {teacher_metrics['tokens_per_sec']:<20.2f} {student_metrics['tokens_per_sec']:<20.2f} {student_metrics['tokens_per_sec']/teacher_metrics['tokens_per_sec']:<10.2f}x")
    print(f"{'Parameters':<25} {teacher_params:<20,} {student_params:<20,} {teacher_params/student_params:<10.2f}x")
    
    print(f"\n{'='*70}")
    print(f"SAMPLE OUTPUTS")
    print(f"{'='*70}\n")
    
    print("TEACHER OUTPUT:")
    print(teacher_metrics['sample_output'][:500])
    print("\n" + "-"*70 + "\n")
    
    print("STUDENT OUTPUT:")
    print(student_metrics['sample_output'][:500])
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    compare_models(num_runs=10)
