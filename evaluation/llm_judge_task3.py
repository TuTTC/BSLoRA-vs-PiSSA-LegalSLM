"""
LLM-as-a-Judge Evaluation for Task 3 (Syllogism Questions)
============================================================
Sử dụng Groq API (llama-3.3-70b-versatile) làm "giám khảo" đánh giá chất lượng
lập luận pháp lý so với ground truth.

Hỗ trợ 2 chế độ:
  1. Groq API  (mặc định, miễn phí cho llama-3.3-70b-versatile)
  2. Local inference  (--use_local, cần ~20GB VRAM)

Usage:
    # Via Groq API (cần set GROQ_API_KEY)
    python evaluation/llm_judge_task3.py \
        --input_file pissa_task3_detailed_non_thinking.json \
        --output_file evaluation/results/judge_non_thinking.json

    # Via local model
    python evaluation/llm_judge_task3.py \
        --input_file pissa_task3_detailed_non_thinking.json \
        --output_file evaluation/results/judge_non_thinking.json \
        --use_local --model_name Qwen/Qwen3-32B-AWQ
"""

import os
import re
import sys
import json
import time
import argparse
import statistics
from typing import Dict, List, Any, Optional
from openai import OpenAI

# ---------------------------------------------------------------------------
# Judge prompt template
# ---------------------------------------------------------------------------
JUDGE_SYSTEM_PROMPT = """\
Bạn là một giám khảo đánh giá chất lượng lập luận pháp lý (legal reasoning).
Bạn sẽ được cung cấp:
  - Câu hỏi pháp lý (Question)
  - Đáp án chuẩn (Reference Answer)
  - Câu trả lời của mô hình (Model Answer)

Nhiệm vụ: Đánh giá câu trả lời của mô hình so với đáp án chuẩn trên 5 tiêu chí theo Thang nhận thức Bloom,
mỗi tiêu chí cho điểm từ 1 đến 5.

Các tiêu chí:
1. recognition (Ghi nhớ): Mô hình có trích xuất và ghi nhớ đúng, chính xác số hiệu điều luật, luật không bịa đặt (hallucinate)?
2. understanding (Đọc hiểu): Mô hình có hiểu đúng ý định của luật và áp dụng vào đúng ngữ cảnh trong câu hỏi không?
3. reasoning (Suy diễn): Cấu trúc lập luận tam đoạn luận có gãy gọn, logic và tiền đề dẫn tới kết luận hợp lý không?
4. interpretation (Giải thích): Có lý giải sâu sắc lý do tại sao áp dụng luật đó không?
5. ethics_bias (Đạo đức/Thiên kiến): Câu trả lời có đảm bảo tính khách quan, hợp đạo đức, không có thiên kiến cá nhân?

Trả lời CHÍNH XÁC theo format JSON sau (không có text thêm):
{
  "recognition": <1-5>,
  "understanding": <1-5>,
  "reasoning": <1-5>,
  "interpretation": <1-5>,
  "ethics_bias": <1-5>,
  "rationale": "<giải thích ngắn gọn bằng tiếng Việt>"
}"""

JUDGE_USER_TEMPLATE = """\
### Câu hỏi (Question):
{question}

### Đáp án chuẩn (Reference Answer):
{reference}

### Câu trả lời của mô hình (Model Answer):
{model_answer}

Hãy đánh giá câu trả lời của mô hình theo 5 tiêu chí trên."""


# ---------------------------------------------------------------------------
# Utility: strip <think>...</think> tags
# ---------------------------------------------------------------------------
def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from model responses."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()


# ---------------------------------------------------------------------------
# Load data (handles both JSON schemas)
# ---------------------------------------------------------------------------
def load_data(input_file: str) -> List[Dict[str, str]]:
    """
    Load and normalize data from either schema:
      - non_thinking: {input, reference, model_answer}
      - thinking:     {user_input, reference, model_response}
    Returns list of dicts with keys: question, reference, model_answer
    """
    with open(input_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    samples = []
    for item in raw_data:
        question = item.get("input") or item.get("user_input", "")
        reference = item.get("reference", "")
        model_answer = item.get("model_answer") or item.get("model_response", "")

        # Strip think tags if present
        model_answer = strip_think_tags(model_answer)

        samples.append({
            "question": question,
            "reference": reference,
            "model_answer": model_answer,
            "original_id": item.get("id", "N/A"),
        })

    return samples


# ---------------------------------------------------------------------------
# API-based judge (OpenRouter / any OpenAI-compatible endpoint)
# ---------------------------------------------------------------------------
def judge_via_api(
    question: str,
    reference: str,
    model_answer: str,
    api_key: str,
    base_url: str = "https://api.groq.com/openai/v1",
    model_name: str = "llama-3.3-70b-versatile",
    max_retries: int = 3,
) -> Dict[str, Any]:
    """Call the judge model via Groq API using OpenAI SDK."""
    
    # Initialize Groq client via OpenAI SDK
    client = OpenAI(api_key=api_key, base_url=base_url)

    user_message = JUDGE_USER_TEMPLATE.format(
        question=question,
        reference=reference,
        model_answer=model_answer,
    )

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.1,
                max_tokens=512,
                response_format={"type": "json_object"},
            )
            
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response content from API")
            
            # Parse JSON from response
            scores = parse_judge_response(content)
            return scores

        except Exception as e:
            error_details = str(e)
            
            # Tự động trích xuất thời gian chờ từ thông báo lỗi Rate Limit của Groq
            # (Ví dụ: "Please try again in 14.2s.")
            wait_match = re.search(r"try again in ([0-9.]+)s", error_details)
            if wait_match:
                wait_time = float(wait_match.group(1)) + 2.0  # Cộng thêm 2s buffer cho chắc chắn
                print(f"  [RATE LIMIT] Đụng trần Groq API. Chờ {wait_time:.1f}s trước khi thử lại ({attempt+1}/{max_retries})...")
            else:
                wait_time = 15 * (attempt + 1)
                print(f"  [RETRY {attempt+1}/{max_retries}] Lỗi: {error_details[:150]}... "
                      f"Chờ {wait_time}s...")
            
            time.sleep(wait_time)

    return {
        "recognition": 0, "understanding": 0,
        "reasoning": 0, "interpretation": 0, "ethics_bias": 0,
        "rationale": f"API call failed after {max_retries} retries",
        "error": True,
    }


# ---------------------------------------------------------------------------
# Local judge (transformers)
# ---------------------------------------------------------------------------
def load_local_judge(model_name: str = "casperhansen/llama-3-70b-instruct-awq"):
    """Load Qwen3-32B-AWQ locally for judging."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[JUDGE] Loading local model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("[JUDGE] Local model loaded successfully.")
    return model, tokenizer


def judge_via_local(
    question: str,
    reference: str,
    model_answer: str,
    model,
    tokenizer,
) -> Dict[str, Any]:
    """Run judge inference locally."""
    import torch

    user_message = JUDGE_USER_TEMPLATE.format(
        question=question,
        reference=reference,
        model_answer=model_answer,
    )

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    return parse_judge_response(response)


# ---------------------------------------------------------------------------
# Parse judge JSON response
# ---------------------------------------------------------------------------
def parse_judge_response(response_text: str) -> Dict[str, Any]:
    """Parse the JSON scores from the judge's response."""
    # Try direct JSON parse
    try:
        scores = json.loads(response_text.strip())
        return validate_scores(scores)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON block from markdown code fence
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if json_match:
        try:
            scores = json.loads(json_match.group(1))
            return validate_scores(scores)
        except json.JSONDecodeError:
            pass

    # Try finding JSON object pattern
    json_match = re.search(r"\{[^{}]*\"legal_accuracy\"[^{}]*\}", response_text, re.DOTALL)
    if json_match:
        try:
            scores = json.loads(json_match.group(0))
            return validate_scores(scores)
        except json.JSONDecodeError:
            pass

    # Fallback: try to extract individual scores via regex
    fallback = {}
    for dim in ["recognition", "understanding", "reasoning",
                 "interpretation", "ethics_bias"]:
        match = re.search(rf'"{dim}"\s*:\s*(\d)', response_text)
        if match:
            fallback[dim] = int(match.group(1))

    if len(fallback) == 5:
        fallback["rationale"] = "Parsed via regex fallback"
        return validate_scores(fallback)

    return {
        "recognition": 0, "understanding": 0,
        "reasoning": 0, "interpretation": 0, "ethics_bias": 0,
        "rationale": "Failed to parse judge response",
        "raw_response": response_text[:500],
        "error": True,
    }


def validate_scores(scores: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure all score dimensions exist and are in valid range [1, 5]."""
    dims = ["recognition", "understanding", "reasoning",
            "interpretation", "ethics_bias"]
    for dim in dims:
        val = scores.get(dim, 0)
        if isinstance(val, (int, float)):
            scores[dim] = max(1, min(5, int(val)))
        else:
            scores[dim] = 0
    if "rationale" not in scores:
        scores["rationale"] = ""
    return scores


# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------
def compute_aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate statistics across all judged samples."""
    dims = ["recognition", "understanding", "reasoning",
            "interpretation", "ethics_bias"]

    valid_results = [r for r in results if not r.get("error", False)]
    n_total = len(results)
    n_valid = len(valid_results)
    n_errors = n_total - n_valid

    agg = {
        "num_samples": n_total,
        "num_evaluated": n_valid,
        "num_errors": n_errors,
    }

    if n_valid == 0:
        for dim in dims:
            agg[f"avg_{dim}"] = 0.0
            agg[f"std_{dim}"] = 0.0
        agg["avg_total"] = 0.0
        return agg

    for dim in dims:
        values = [r[dim] for r in valid_results if r[dim] > 0]
        if values:
            agg[f"avg_{dim}"] = round(statistics.mean(values), 3)
            agg[f"std_{dim}"] = round(statistics.stdev(values), 3) if len(values) > 1 else 0.0
        else:
            agg[f"avg_{dim}"] = 0.0
            agg[f"std_{dim}"] = 0.0

    # Average of all 5 dimension averages
    dim_avgs = [agg[f"avg_{dim}"] for dim in dims if agg[f"avg_{dim}"] > 0]
    agg["avg_total"] = round(statistics.mean(dim_avgs), 3) if dim_avgs else 0.0

    return agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="LLM-as-a-Judge evaluation for Task 3 (Syllogism Questions)"
    )
    parser.add_argument(
        "--input_file", type=str, required=True,
        help="Path to JSON file with model responses "
             "(pissa_task3_detailed_non_thinking.json or "
             "pissa_task3_detailed_responses_thinking.json)"
    )
    parser.add_argument(
        "--output_file", type=str, default=None,
        help="Path to save judge results JSON (default: auto-generated)"
    )
    parser.add_argument(
        "--num_samples", type=int, default=-1,
        help="Number of samples to evaluate (-1 = all)"
    )
    parser.add_argument(
        "--use_local", action="store_true",
        help="Use local Qwen3-32B-AWQ instead of API"
    )
    parser.add_argument(
        "--use_openrouter", action="store_true",
        help="Use OpenRouter API (default if --use_local not specified)"
    )
    parser.add_argument(
        "--model_name", type=str, default="casperhansen/llama-3-70b-instruct-awq",
        help="Model name for local inference or API model identifier"
    )
    parser.add_argument(
        "--api_model", type=str, default="llama-3.3-70b-versatile",
        help="Model name for Groq API (suitable for legal reasoning)"
    )
    parser.add_argument(
        "--api_base_url", type=str, default="https://api.groq.com/openai/v1",
        help="Base URL for Groq API (OpenAI-compatible)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing output file (skip already-judged samples)"
    )
    parser.add_argument(
        "--retry_errors", action="store_true",
        help="Re-evaluate samples that failed (error=true) in the existing output file, then exit"
    )
    parser.add_argument(
        "--print_summary", action="store_true",
        help="Just print the aggregate summary of an existing output JSON file and exit"
    )
    args = parser.parse_args()

    # 0. Print summary only
    if args.print_summary:
        if not args.output_file or not os.path.exists(args.output_file):
            print(f"[ERROR] Cannot find output file to summarize: {args.output_file}")
            sys.exit(1)
            
        with open(args.output_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            results = data.get("per_sample_results", [])
            
        agg = compute_aggregate(results)
        print(f"\n{'='*60}")
        print(f"  EVALUATION SUMMARY: {os.path.basename(args.output_file)}")
        print(f"{'='*60}")
        print(f"  Samples evaluated: {agg['num_evaluated']}/{agg['num_samples']}")
        print(f"  Errors:            {agg['num_errors']}")
        print(f"{'─'*60}")
        print(f"  Recognition (L1):  {agg.get('avg_recognition', 0):.2f} ± {agg.get('std_recognition', 0):.2f}")
        print(f"  Understanding(L2): {agg.get('avg_understanding', 0):.2f} ± {agg.get('std_understanding', 0):.2f}")
        print(f"  Reasoning (L3):    {agg.get('avg_reasoning', 0):.2f} ± {agg.get('std_reasoning', 0):.2f}")
        print(f"  Interpretation(L4):{agg.get('avg_interpretation', 0):.2f} ± {agg.get('std_interpretation', 0):.2f}")
        print(f"  Ethics/Bias (L5):  {agg.get('avg_ethics_bias', 0):.2f} ± {agg.get('std_ethics_bias', 0):.2f}")
        print(f"{'─'*60}")
        print(f"  AVERAGE TOTAL:     {agg.get('avg_total', 0):.2f}")
        print(f"{'='*60}")
        sys.exit(0)

    # 1. Load data
    print(f"\n{'='*60}")
    print(f"  LLM-as-a-Judge — Task 3 Evaluation")
    print(f"{'='*60}")
    print(f"[DATA] Loading: {args.input_file}")

    samples = load_data(args.input_file)
    if args.num_samples > 0:
        samples = samples[:args.num_samples]
    print(f"[DATA] Loaded {len(samples)} samples")

    # 2. Determine output file
    if args.output_file is None:
        basename = os.path.splitext(os.path.basename(args.input_file))[0]
        output_dir = os.path.join(os.path.dirname(args.input_file), "evaluation", "results")
        os.makedirs(output_dir, exist_ok=True)
        args.output_file = os.path.join(output_dir, f"judge_{basename}.json")

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 3. Resume support
    existing_results = []
    start_idx = 0
    if (args.resume or args.retry_errors) and os.path.exists(args.output_file):
        with open(args.output_file, "r", encoding="utf-8") as f:
            saved = json.load(f)
            existing_results = saved.get("per_sample_results", [])
            if args.resume:
                start_idx = len(existing_results)
                print(f"[RESUME] Found {start_idx} existing results, continuing...")
            if args.retry_errors:
                print(f"[RETRY] Loaded {len(existing_results)} existing results to retry errors.")

    # 4. Setup judge
    local_model = None
    local_tokenizer = None

    if args.use_local:
        local_model, local_tokenizer = load_local_judge(args.model_name)
    else:
        # Get API key from environment variable
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            print("[ERROR] GROQ_API_KEY environment variable not set!")
            print("\n  Get your free API key from: https://console.groq.com/keys")
            print("\n  Set your API key in PowerShell:")
            print("    $env:GROQ_API_KEY = 'your-groq-api-key-here'")
            print("\n  Or in bash:")
            print("    export GROQ_API_KEY='your-groq-api-key-here'")
            print("\n  Or use --use_local for local inference")
            sys.exit(1)
        print(f"[API] Using Groq API with model: {args.api_model}")
        print(f"[API] Base URL: {args.api_base_url}")

    # 5. Run evaluation
    results = existing_results.copy()

    # Nếu người dùng muốn đánh giá lại các case bị lỗi
    if args.retry_errors:
        print(f"\n[RETRY] Re-evaluating error samples...")
        print(f"{'─'*60}")
        error_indices = [i for i, r in enumerate(results) if r.get("error", False)]
        print(f"  Found {len(error_indices)} samples with error=true.")
        
        for i in error_indices:
            sample = samples[i]
            print(f"  [RETRY] Judging sample {sample['original_id']} (Index {i})...", end=" ", flush=True)

            if args.use_local:
                scores = judge_via_local(
                    question=sample["question"],
                    reference=sample["reference"],
                    model_answer=sample["model_answer"],
                    model=local_model,
                    tokenizer=local_tokenizer,
                )
            else:
                scores = judge_via_api(
                    question=sample["question"],
                    reference=sample["reference"],
                    model_answer=sample["model_answer"],
                    api_key=api_key,
                    base_url=args.api_base_url,
                    model_name=args.api_model,
                )

            # Add metadata
            scores["sample_id"] = sample["original_id"]
            scores["sample_index"] = i

            is_error = scores.get("error", False)
            if is_error:
                print("❌ (error)")
            else:
                # Tính điểm TB ngay lúc in
                dim_vals = [scores.get(d, 0) for d in ["recognition", "understanding", "reasoning", "interpretation", "ethics_bias"]]
                overall = round(sum(dim_vals) / 5, 2)
                print(f"✓ (overall: {overall}/5)")

            results[i] = scores

            # Save checkpoint
            agg = compute_aggregate(results)
            output_data = {
                "input_file": args.input_file,
                "judge_model": args.api_model if not args.use_local else args.model_name,
                "aggregate_scores": agg,
                "per_sample_results": results,
            }
            with open(args.output_file, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            # Rate limiting for Groq API
            if not args.use_local:
                time.sleep(6)
                
        print(f"\n[RETRY] Done re-evaluating {len(error_indices)} errors.")
        sys.exit(0)

    print(f"\n[EVAL] Evaluating {len(samples) - start_idx} new samples...")
    print(f"{'─'*60}")

    for i, sample in enumerate(samples[start_idx:], start=start_idx):
        print(f"  [{i+1}/{len(samples)}] Judging sample {sample['original_id']}...", end=" ", flush=True)

        if args.use_local:
            scores = judge_via_local(
                question=sample["question"],
                reference=sample["reference"],
                model_answer=sample["model_answer"],
                model=local_model,
                tokenizer=local_tokenizer,
            )
        else:
            scores = judge_via_api(
                question=sample["question"],
                reference=sample["reference"],
                model_answer=sample["model_answer"],
                api_key=api_key,
                base_url=args.api_base_url,
                model_name=args.api_model,
            )

        # Add metadata
        scores["sample_id"] = sample["original_id"]
        scores["sample_index"] = i

        is_error = scores.get("error", False)
        if is_error:
            print("❌ (error)")
        else:
            # Tính điểm TB ngay lúc in
            dim_vals = [scores.get(d, 0) for d in ["recognition", "understanding", "reasoning", "interpretation", "ethics_bias"]]
            overall = round(sum(dim_vals) / 5, 2)
            print(f"✓ (overall: {overall}/5)")

        results.append(scores)

        # Checkpoint: save every 10 samples
        if (i + 1) % 10 == 0 or (i + 1) == len(samples):
            agg = compute_aggregate(results)
            output_data = {
                "input_file": args.input_file,
                "judge_model": args.api_model if not args.use_local else args.model_name,
                "aggregate_scores": agg,
                "per_sample_results": results,
            }
            with open(args.output_file, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

        # Rate limiting for Groq API (stricter RPM limits for free tier)
        if not args.use_local:
            time.sleep(6)

    # 6. Final summary
    agg = compute_aggregate(results)
    print(f"\n{'='*60}")
    print(f"  EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"  Samples evaluated: {agg['num_evaluated']}/{agg['num_samples']}")
    print(f"  Errors:            {agg['num_errors']}")
    print(f"{'─'*60}")
    print(f"  Recognition (L1):  {agg['avg_recognition']:.2f} ± {agg['std_recognition']:.2f}")
    print(f"  Understanding(L2): {agg['avg_understanding']:.2f} ± {agg['std_understanding']:.2f}")
    print(f"  Reasoning (L3):    {agg['avg_reasoning']:.2f} ± {agg['std_reasoning']:.2f}")
    print(f"  Interpretation(L4):{agg['avg_interpretation']:.2f} ± {agg['std_interpretation']:.2f}")
    print(f"  Ethics/Bias (L5):  {agg['avg_ethics_bias']:.2f} ± {agg['std_ethics_bias']:.2f}")
    print(f"{'─'*60}")
    print(f"  AVERAGE TOTAL:     {agg['avg_total']:.2f}")
    print(f"{'='*60}")

    # Save final results
    output_data = {
        "input_file": args.input_file,
        "judge_model": args.api_model if not args.use_local else args.model_name,
        "aggregate_scores": agg,
        "per_sample_results": results,
    }
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVE] Results saved to: {args.output_file}")

if __name__ == "__main__":
    main()
