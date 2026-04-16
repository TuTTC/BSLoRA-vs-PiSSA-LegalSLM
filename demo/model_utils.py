# demo/model_utils.py
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Khai báo sẵn các đường dẫn (Sửa lại cho khớp với folder của bạn)
CHECKPOINTS = {
    "LoRA+ (Non-Thinking)": "outputs/checkpoints/loraplus_resumed/checkpoint-700",
    "LoRA+ (Thinking)": "outputs/checkpoints/loraplus_thinking/best_model",
    "PiSSA (Non-Thinking)": "outputs/checkpoints/pissa",
    "PiSSA (Thinking)": "outputs/checkpoints/pissa_thinking/best_model",
}

BASE_MODEL_PATH = "VLSP2025-LegalSML/qwen3-4b-legal-pretrain" # Sửa thành model gốc bạn đang dùng

current_model = None
current_tokenizer = None
def clear_vram():
    global current_model, current_tokenizer
    if current_model is not None: del current_model
    if current_tokenizer is not None: del current_tokenizer
    gc.collect()
    torch.cuda.empty_cache()

def load_selected_model(model_choice):
    global current_model, current_tokenizer
    try:
        clear_vram()
        adapter_path = CHECKPOINTS[model_choice]
        current_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
        current_tokenizer.padding_side = "left"
        
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH, 
            torch_dtype=torch.bfloat16, 
            device_map="auto"
        )
        current_model = PeftModel.from_pretrained(base_model, adapter_path)
        current_model.eval()
        return f"✅ Đã sẵn sàng: {model_choice}"
    except Exception as e:
        return f"❌ Lỗi hệ thống: {str(e)}"

def generate_answer(system_prompt: str, user_input: str):
    global current_model, current_tokenizer
    if current_model is None:
        return "⚠️ Vui lòng nạp mô hình trước."
        
    prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_input}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    inputs = current_tokenizer(prompt, return_tensors="pt").to(current_model.device)

    with torch.no_grad():
        outputs = current_model.generate(
            **inputs,
            max_new_tokens=512,  # SET CỨNG
            temperature=0.1,     # SET CỨNG
            top_p=0.9,           # SET CỨNG
            do_sample=True,
        )

    response = current_tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return response.strip()