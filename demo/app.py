"""
🇻🇳 Legal QA Chatbot - PiSSA Fine-tuned Qwen3-4B
Run locally with: python demo/app.py
"""

import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ==============================================================================
# Configuration
# ==============================================================================
BASE_MODEL_ID = "VLSP2025-LegalSML/qwen3-4b-legal-pretrain"
ADAPTER_PATH = "./outputs-final/outputs/checkpoints/pissa"  # local path

# ==============================================================================
# Load Model on CPU (safe for laptops with limited VRAM)
# ==============================================================================
print(" Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)

print(" Loading base model on CPU (this may take 1-2 minutes)...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float32,
    device_map="cpu",
    trust_remote_code=True,
)

print(" Loading PiSSA adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()
print(" Model loaded successfully! (Running on CPU — responses may take 30-60s)")


# ==============================================================================
# Inference
# ==============================================================================
def respond(message, history, system_message, max_tokens, temperature, top_p):
    """Generate a response from the model."""
    messages = [{"role": "system", "content": system_message}]

    for user_msg, bot_msg in history:
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if bot_msg:
            messages.append({"role": "assistant", "content": bot_msg})

    messages.append({"role": "user", "content": message})

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=int(max_tokens),
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            repetition_penalty=1.1,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )

    # Remove <think>...</think> blocks if present
    if "</think>" in response:
        response = response.split("</think>")[-1].strip()

    return response


# ==============================================================================
# Gradio UI
# ==============================================================================
demo = gr.ChatInterface(
    fn=respond,
    additional_inputs=[
        gr.Textbox(
            value="Bạn là một trợ lý pháp luật Việt Nam. Hãy trả lời câu hỏi một cách chính xác và dễ hiểu.",
            label="System Prompt",
        ),
        gr.Slider(64, 2048, value=512, step=64, label="Max Tokens"),
        gr.Slider(0.1, 1.5, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p"),
    ],
    title="🇻🇳 Hỏi Đáp Pháp Luật Việt Nam",
    description=(
        "**PiSSA Fine-tuned Qwen3-4B** trên dữ liệu pháp luật Việt Nam.\n\n"
        "Nhập câu hỏi pháp luật bằng tiếng Việt để nhận câu trả lời."
    ),
    examples=[
        ["Thủ tục đăng ký kết hôn gồm những bước nào?"],
        ["Quyền và nghĩa vụ của người lao động theo Bộ luật Lao động?"],
        ["Hợp đồng lao động có thời hạn tối đa bao lâu?"],
        ["Điều kiện để được hưởng bảo hiểm thất nghiệp là gì?"],
    ],
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
