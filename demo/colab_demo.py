##############################################################################
# 🇻🇳 Legal QA Demo — Google Colab Notebook
# Copy this into a Colab notebook (one cell per section)
# Make sure to select: Runtime > Change runtime type > T4 GPU
##############################################################################

# ============================================================================
# CELL 1: Install dependencies
# ============================================================================
# !pip install -q torch transformers peft accelerate bitsandbytes gradio safetensors

# ============================================================================
# CELL 2: Load model with 4-bit quantization (fits in Colab T4 16GB VRAM)
# ============================================================================
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from peft import PeftModel
from threading import Thread

# ⚠️ CHANGE THIS to your HF repo (after running upload_adapter_to_hf.py)
BASE_MODEL_ID = "VLSP2025-LegalSML/qwen3-4b-legal-pretrain"
ADAPTER_ID = "camtus/qwen3-4b-pissa-legal"

print(" Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_ID, trust_remote_code=True)

print(" Loading base model in float16 (~8GB, fits in T4 16GB VRAM)...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

print(" Loading PiSSA adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_ID)
model.eval()
print(" Model loaded successfully!")

# ============================================================================
# CELL 3: Define inference function (Streaming Mode - No more 504 Timeouts!)
# ============================================================================
def respond(message, history, system_message, max_tokens, temperature, top_p):
    """Generate a streaming response from the model."""
    messages = [{"role": "system", "content": system_message}]
    for msg in history:
        messages.append(msg)
    messages.append({"role": "user", "content": message})

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # Initialize streamer
    streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)

    generate_kwargs = dict(
        **inputs,
        max_new_tokens=int(max_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
        do_sample=True,
        repetition_penalty=1.1,
        streamer=streamer,
    )

    # Run generation in a separate thread
    thread = Thread(target=model.generate, kwargs=generate_kwargs)
    thread.start()

    # Yield tokens one by one (this keeps the connection alive)
    generated_text = ""
    for new_text in streamer:
        generated_text += new_text
        # Remove <think> blocks if they appear during streaming
        if "</think>" in generated_text:
             yield generated_text.split("</think>")[-1].strip()
        else:
             yield generated_text

# ============================================================================
# CELL 4: Launch Gradio chatbot (share=True gives you a public URL)
# ============================================================================
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
    type="messages",  # Use modern message format
)

demo.launch(debug=True, share=True)
