# demo/prompts.py

SYSTEM_PROMPTS = {
    "task1": (
        "Bạn là một chuyên gia pháp luật Việt Nam. "
        "Nhiệm vụ của bạn là xác định xem một điều luật "
        "có thể được sử dụng để trả lời câu hỏi pháp lý cụ thể hay không."
    ),
    "task2": (
        "Bạn là một chuyên gia pháp luật Việt Nam. "
        "Hãy trả lời câu hỏi trắc nghiệm sau dựa trên "
        "văn bản pháp luật được cung cấp."
    ),
    "task3": (
        "Bạn là một chuyên gia pháp luật Việt Nam. "
        "Hãy trả lời câu hỏi mở sau theo cấu trúc "
        "lập luận pháp lý chuyên sâu."
    ),
}


def get_system_prompt(task_type: str, is_thinking_mode: bool = False) -> str:
    """
    Trả về system prompt gốc. 
    Tham số is_thinking_mode được giữ lại để không làm lỗi app.py, 
    nhưng sẽ không cộng thêm bất kỳ chuỗi <think> nào nữa.
    """
    return SYSTEM_PROMPTS.get(task_type, SYSTEM_PROMPTS["task3"])