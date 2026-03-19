import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

checkpoint_path = "outputs/checkpoints/pissa"
base_model_name = "VLSP2025-LegalSML/qwen3-4b-legal-pretrain"

print(f"Loading Base Model...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

print(f"Loading PiSSA Adapter...")
model = PeftModel.from_pretrained(model, checkpoint_path)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
all_params = sum(p.numel() for p in model.parameters())

print("\n" + "="*40)
print(f"MODEL PARAMETERS REPORT")
print("="*40)
print(f"Trainable params: {trainable_params:,}")
print(f"All params:       {all_params:,}")
print(f"Trainable %:      {100 * trainable_params / all_params:.4f}%")
print("="*40)
