import torch
from model import DiffusionTransformer, DiffusionConfig, decode_tokens

def main():
    # ======= Student Model Config (adjust if different) =======
    student_config = DiffusionConfig(
        n_layer=5,
        n_head=4,
        n_embd=128,
        # KEEP all other parameters the same as your student distillation (vocab_size, sequence_len, etc)
    )

    # ======= Load student model =======
    student = DiffusionTransformer(student_config)
    student.load_state_dict(torch.load("weights/student_1M.pt", map_location="cpu"))
    student.eval()

    # ======= Sampling =======
    with torch.no_grad():
        # Generates a sequence. Adjust batch_size, seq_len as desired.
        samples = student.sample(
            batch_size=1,
            seq_len=student_config.sequence_len,   # e.g., 256
            device="cpu",
            method="confidence",                  # Or your sampling strategy
            confidence_threshold=0.95              # Adjust as needed
        )
        # If text model: decode
        print("\n--- Generated Output ---\n")
        print(decode_tokens(samples[0]))
        print("\n----------------------\n")

if __name__ == "__main__":
    main()
