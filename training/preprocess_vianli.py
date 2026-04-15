"""
ViANLI Preprocessing Utility
============================
Formats the uitnlp/ViANLI dataset into the specified template for Qwen2.5.
"""

def map_label(label_id):
    """Maps integer labels to text labels for ViANLI."""
    mapping = {
        0: "entailment",
        1: "neutral",
        2: "contradiction"
    }
    return mapping.get(label_id, "unknown")

def format_vianli_prompt(example, tokenizer, template):
    """
    Formats a single example into the prompt template.
    Template: User: "Câu 1: [premise]. Câu 2: [hypothesis]. Mối quan hệ giữa hai câu là gì?"
              Assistant: "[label_text]"
    """
    premise = example.get("premise", "")
    hypothesis = example.get("hypothesis", "")
    label_id = example.get("label", -1)
    label_text = map_label(label_id)
    
    # Fill template
    formatted_text = template.format(
        premise=premise,
        hypothesis=hypothesis,
        label_text=label_text
    )
    
    # Check for EOS token in Unsloth style (optional, handled by trainer usually)
    # but here we return the raw text to be tokenized by the trainer.
    return formatted_text

def get_vianli_formatter(tokenizer, template):
    def formatter(examples):
        texts = []
        for i in range(len(examples["premise"])):
            ex = {
                "premise": examples["premise"][i],
                "hypothesis": examples["hypothesis"][i],
                "label": examples["label"][i]
            }
            texts.append(format_vianli_prompt(ex, tokenizer, template) + tokenizer.eos_token)
        return {"text": texts}
    return formatter
