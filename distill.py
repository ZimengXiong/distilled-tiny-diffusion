import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from model import DiffusionTransformer, DiffusionConfig, decode_tokens
from training import MaskedDiffusionSchedule, get_data_loader 

student_config = DiffusionConfig(n_layer=5, n_head=4, n_embd=128)

TEACHER_PATH = "weights/diffusion_model.pt"
STUDENT_PATH = "weights/student_1M.pt"
CHECKPOINT_PATH = "weights/checkpoint_student.pt"
DATA_PATH = "data/tiny_shakespeare.txt"

BATCH_SIZE = 64
NUM_STEPS = 10000
LEARNING_RATE = 5e-4
TEMPERATURE = 2.0
ALPHA = 0.7

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))
print(f"Using device: {device}")

def distillation_loss(teacher_logits, student_logits, targets, mask, temperature=2.0, alpha=0.7):
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    kl_loss = F.kl_div(
        student_log_probs, teacher_probs, reduction='none').sum(dim=-1)
    kl_loss = (kl_loss * mask).sum() / mask.sum()
    kl_loss *= (temperature ** 2)
    ce_loss = F.cross_entropy(
        student_logits.reshape(-1, student_logits.size(-1)),
        targets.reshape(-1),
        reduction='none'
    )
    ce_loss = (ce_loss.view_as(mask) * mask).sum() / mask.sum()
    loss = alpha * kl_loss + (1 - alpha) * ce_loss
    return loss, kl_loss.item(), ce_loss.item()

def distill_step(teacher, student, x_0, mask_schedule, optimizer, temperature=2.0, alpha=0.7):
    B, _ = x_0.shape
    t = torch.randint(0, mask_schedule.num_timesteps, (B,), device=x_0.device)
    x_t = mask_schedule.add_masks(x_0, t)
    with torch.no_grad():
        teacher_logits = teacher(x_t, t)
    student_logits = student(x_t, t)
    mask = (x_t == mask_schedule.mask_token_id).float()
    loss, kl, ce = distillation_loss(teacher_logits, student_logits, x_0, mask, temperature, alpha)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item(), kl, ce

# --- Resume logic / Main loop ---
def distill_model():
    print("Loading teacher model...")
    teacher_cfg = DiffusionConfig()
    teacher = DiffusionTransformer(teacher_cfg).to(device)
    teacher.load_state_dict(torch.load(TEACHER_PATH, map_location=device))
    teacher.eval()
    teacher_params = sum(p.numel() for p in teacher.parameters())
    print(f"Teacher params: {teacher_params:,}")

    print("Initializing student model...")
    student = DiffusionTransformer(student_config).to(device)
    student.init_weights()
    student_params = sum(p.numel() for p in student.parameters())
    print(f"Student params: {student_params:,}")
    print(f"Compression ratio: {teacher_params / student_params:.1f}x\n")

    mask_schedule = MaskedDiffusionSchedule(
        num_timesteps=student_config.diffusion_steps,
        mask_token_id=student_config.mask_token_id,
        context_len=student_config.context_len,
    )

    data_loader = get_data_loader(DATA_PATH, BATCH_SIZE, student_config.sequence_len, device)

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
        weight_decay=0.01
    )
    from torch.optim.lr_scheduler import OneCycleLR
    scheduler = OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        total_steps=NUM_STEPS,
        pct_start=0.05,
        anneal_strategy='cos'
    )

    start_step = 0
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        student.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_step = checkpoint['step']
        print(f"Resumed from checkpoint at step {start_step}")
    else:
        print("No checkpoint found. Starting fresh.")

    pbar = tqdm(range(start_step, NUM_STEPS), desc="Distilling")
    for step in pbar:
        x_0 = next(data_loader)
        loss, kl, ce = distill_step(
            teacher, student, x_0, mask_schedule, optimizer, TEMPERATURE, ALPHA
        )
        scheduler.step()
        pbar.set_postfix({
            'loss': f'{loss:.4f}',
            'kl': f'{kl:.4f}',
            'ce': f'{ce:.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}'
        })
        if (step + 1) % 1000 == 0:
            student.eval()
            with torch.no_grad():
                samples = student.sample(
                    batch_size=1,
                    seq_len=student_config.sequence_len,
                    device=device,
                    method='confidence',
                    confidence_threshold=0.95
                )
                text = decode_tokens(samples[0])
                tqdm.write(f"\n--- Student sample at step {step+1} ---")
                tqdm.write(text[:200])
                tqdm.write("---\n")
            student.train()
        if (step + 1) % 1000 == 0:
            checkpoint = {
                'model_state_dict': student.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'step': step + 1
            }
            torch.save(checkpoint, CHECKPOINT_PATH)
            torch.save(student.state_dict(), STUDENT_PATH)
    torch.save(student.state_dict(), STUDENT_PATH)
    print(f"\nStudent saved to {STUDENT_PATH}")

if __name__ == "__main__":
    distill_model()
