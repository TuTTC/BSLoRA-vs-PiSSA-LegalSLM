# """
# 🇻🇳 Legal QA Chatbot - PiSSA Fine-tuned Qwen3-4B
# Run locally with: python demo/app.py
# """

# import torch
# import gradio as gr
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from peft import PeftModel

# # ==============================================================================
# # Configuration
# # ==============================================================================
# BASE_MODEL_ID = "VLSP2025-LegalSML/qwen3-4b-legal-pretrain"
# ADAPTER_PATH = "/BSLoRA-vs-PiSSA-LegalSLM/outputs/checkpoints/pissa"  # local path

# # ==============================================================================
# # Load Model on CPU (safe for laptops with limited VRAM)
# # ==============================================================================
# print(" Loading tokenizer...")
# tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)

# print(" Loading base model on CPU (this may take 1-2 minutes)...")
# base_model = AutoModelForCausalLM.from_pretrained(
#     BASE_MODEL_ID,
#     torch_dtype=torch.float32,
#     device_map="cpu",
#     trust_remote_code=True,
# )

# print(" Loading PiSSA adapter...")
# model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
# model.eval()
# print(" Model loaded successfully! (Running on CPU — responses may take 30-60s)")


# # ==============================================================================
# # Inference
# # ==============================================================================
# def respond(message, history, system_message, max_tokens, temperature, top_p):
#     """Generate a response from the model."""
#     messages = [{"role": "system", "content": system_message}]

#     for user_msg, bot_msg in history:
#         if user_msg:
#             messages.append({"role": "user", "content": user_msg})
#         if bot_msg:
#             messages.append({"role": "assistant", "content": bot_msg})

#     messages.append({"role": "user", "content": message})

#     text = tokenizer.apply_chat_template(
#         messages, tokenize=False, add_generation_prompt=True
#     )
#     inputs = tokenizer(text, return_tensors="pt").to(model.device)

#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=int(max_tokens),
#             temperature=temperature,
#             top_p=top_p,
#             do_sample=True,
#             repetition_penalty=1.1,
#         )

#     response = tokenizer.decode(
#         outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
#     )

#     # Remove <think>...</think> blocks if present
#     if "</think>" in response:
#         response = response.split("</think>")[-1].strip()

#     return response


# # ==============================================================================
# # Gradio UI
# # ==============================================================================
# demo = gr.ChatInterface(
#     fn=respond,
#     additional_inputs=[
#         gr.Textbox(
#             value="Bạn là một trợ lý pháp luật Việt Nam. Hãy trả lời câu hỏi một cách chính xác và dễ hiểu.",
#             label="System Prompt",
#         ),
#         gr.Slider(64, 2048, value=512, step=64, label="Max Tokens"),
#         gr.Slider(0.1, 1.5, value=0.7, step=0.1, label="Temperature"),
#         gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p"),
#     ],
#     title="🇻🇳 Hỏi Đáp Pháp Luật Việt Nam",
#     description=(
#         "**PiSSA Fine-tuned Qwen3-4B** trên dữ liệu pháp luật Việt Nam.\n\n"
#         "Nhập câu hỏi pháp luật bằng tiếng Việt để nhận câu trả lời."
#     ),
#     examples=[
#         ["Thủ tục đăng ký kết hôn gồm những bước nào?"],
#         ["Quyền và nghĩa vụ của người lao động theo Bộ luật Lao động?"],
#         ["Hợp đồng lao động có thời hạn tối đa bao lâu?"],
#         ["Điều kiện để được hưởng bảo hiểm thất nghiệp là gì?"],
#     ],
#     theme=gr.themes.Soft(),
# )

# if __name__ == "__main__":
#     demo.launch(server_name="0.0.0.0", server_port=7860, share=True)


# demo/app.py
import gradio as gr
import os
from model_utils import CHECKPOINTS, load_selected_model, generate_answer
from prompts import get_system_prompt
from data_utils import load_sample_data

# Tải dữ liệu test mẫu
sample_data = load_sample_data(num_samples=15)

def process_query(model_choice, task_type, input_text):
    is_thinking = "Thinking" in model_choice
    sys_prompt = get_system_prompt(task_type, is_thinking_mode=is_thinking)
    return generate_answer(sys_prompt, input_text)

# Đọc CSS từ file external để code Python gọn gàng, chuyên nghiệp hơn
css_path = os.path.join(os.path.dirname(__file__), "style.css")
with open(css_path, "r", encoding="utf-8") as f:
    custom_css = f.read()

with gr.Blocks(theme=gr.themes.Default(primary_hue="blue", secondary_hue="slate"), css=custom_css) as demo:
    # Header Section
    with gr.Column(elem_classes="legal-header"):
        gr.Markdown("# ⚖️ LegalSLM: Hệ thống Trợ lý Pháp luật Thông minh")
        gr.Markdown("### *Fine-tuned on Vietnamese Legal Dataset (VLSP 2025)*")

    with gr.Row():
        # Cột Sidebar bên trái (Cấu hình)
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### 🛠️ Thiết lập mô hình")
                model_dropdown = gr.Dropdown(
                    choices=list(CHECKPOINTS.keys()), 
                    label="Chọn Checkpoint", 
                    value=list(CHECKPOINTS.keys())[0]
                )
                load_btn = gr.Button("🚀 Nạp trọng số mô hình", variant="primary")
                load_status = gr.Textbox(label="Trạng thái hệ thống", interactive=False, placeholder="Đợi nạp...")
            
            with gr.Accordion("ℹ️ Thông tin kỹ thuật", open=False):
                gr.Markdown("""
                - **Kiến trúc:** Qwen 3-4B
                - **Phương pháp:** LoRA+ / PiSSA
                - **Tham số sinh:** Fixed (Tokens: 512, Temp: 0.1)
                """)

        # Cột chính bên phải (Workspace)
        with gr.Column(scale=3):
            with gr.Tabs():
                # --- TASK 1 ---
                with gr.TabItem("📋 TASK 1: Xác định Điều Luật"):
                    with gr.Group(elem_classes="task-box"):
                        gr.Markdown("#### 📥 Chọn mẫu thử từ Public Test")
                        # Đã xóa lỗi "placeholder" tại Dropdown
                        t1_sample = gr.Dropdown(choices=sample_data["task1"], label="")
                        
                        t1_input = gr.Textbox(lines=6, label="Nội dung phân tích", placeholder="Nhập câu hỏi và điều luật...")
                        t1_sample.change(lambda x: x, inputs=t1_sample, outputs=t1_input)
                        
                        t1_btn = gr.Button("🔍 Kiểm tra tính liên quan", variant="primary")
                        t1_output = gr.Textbox(lines=3, label="Kết quả dự đoán", interactive=False)
                        t1_btn.click(fn=lambda m, i: process_query(m, "task1", i), inputs=[model_dropdown, t1_input], outputs=t1_output)

                # --- TASK 2 ---
                with gr.TabItem("📝 TASK 2: Trắc nghiệm Pháp luật"):
                    with gr.Group(elem_classes="task-box"):
                        gr.Markdown("#### 📥 Chọn câu hỏi mẫu")
                        t2_sample = gr.Dropdown(choices=sample_data["task2"], label="")
                        
                        t2_input = gr.Textbox(lines=7, label="Câu hỏi & Lựa chọn", placeholder="Nội dung câu hỏi MCQ...")
                        t2_sample.change(lambda x: x, inputs=t2_sample, outputs=t2_input)
                        
                        t2_btn = gr.Button("🎯 Giải đáp câu hỏi", variant="primary")
                        t2_output = gr.Textbox(lines=2, label="Đáp án lựa chọn", interactive=False)
                        t2_btn.click(fn=lambda m, i: process_query(m, "task2", i), inputs=[model_dropdown, t2_input], outputs=t2_output)

                # --- TASK 3 ---
                with gr.TabItem("📖 TASK 3: Lập luận Pháp lý"):
                    with gr.Group(elem_classes="task-box"):
                        gr.Markdown("#### 📥 Chọn tình huống mẫu")
                        t3_sample = gr.Dropdown(choices=sample_data["task3"], label="")
                        
                        t3_input = gr.Textbox(lines=5, label="Tình huống pháp lý", placeholder="Nhập câu hỏi mở cần lập luận...")
                        t3_sample.change(lambda x: x, inputs=t3_sample, outputs=t3_input)
                        
                        t3_btn = gr.Button("💡 Tạo văn bản lập luận", variant="primary")
                        t3_output = gr.Textbox(lines=12, label="Phân tích chi tiết", interactive=False)
                        t3_btn.click(fn=lambda m, i: process_query(m, "task3", i), inputs=[model_dropdown, t3_input], outputs=t3_output)

    load_btn.click(fn=load_selected_model, inputs=model_dropdown, outputs=load_status)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)