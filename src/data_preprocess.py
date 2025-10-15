import json
from typing import List, Dict, Any
from transformers import AutoTokenizer


def load_questions(filename: str) -> List[Dict[str, Any]]:
    """Load questions from a JSON or JSONL file."""
    with open(filename, "r") as f:
        if filename.endswith(".jsonl"):
            questions = [json.loads(line) for line in f]
        else:
            questions = json.load(f)
    return questions


_tokenizer_cache = {}


def get_tokenizer(model: str) -> AutoTokenizer:
    """Get or load a tokenizer with caching."""
    if model not in _tokenizer_cache:
        _tokenizer_cache[model] = AutoTokenizer.from_pretrained(model)
    return _tokenizer_cache[model]


def create_prompts(
    questions: List[Dict[str, Any]], model_name: str, use_agent: bool = False
) -> List[str]:
    """Create prompts for a batch of questions."""
    prompts = []

    for question_data in questions:
        question = question_data["question"]
        options = question_data.get("options", {})

        # Parse options if they're a string
        if isinstance(options, str):
            try:
                options = json.loads(options)
            except (json.JSONDecodeError, ValueError):
                options = {}

        # Create the options text
        options_text = ""
        for key, value in options.items():
            options_text += f"{key}: {value}\n"

        # Determine the valid options
        valid_options = list(options.keys())
        options_list = ", ".join(valid_options)

        if use_agent:
            # For agent mode: return simple, clean question text
            prompt = f"""Question: {question}

Options:
{options_text}

INSTRUCTIONS: If this question involves cancer/genes/proteins, USE TOOLS IMMEDIATELY. Answer with <answer>[letter]</answer>"""
        else:
            # For direct completion: use tokenizer chat template
            tokenizer = get_tokenizer(model_name)
            messages = [
                {
                    "role": "system",
                    "content": f"You are a biology expert. Answer the following multiple choice questions by selecting the correct option ({options_list}) and providing a brief explanation. Always format your answer as <answer>[letter]</answer>.",
                },
                {
                    "role": "user",
                    "content": f"""Question: {question}

Options:
{options_text}
Please provide your answer as a single letter ({options_list}).
Format your answer as: <answer>[letter]</answer>

Answer:""",
                },
            ]

            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        prompts.append(prompt)

    return prompts


def make_batches(
    questions: List[Dict[str, Any]], batch_size: int = 8
) -> List[List[Dict[str, Any]]]:
    """Split questions into batches for processing."""
    batches = []
    for i in range(0, len(questions), batch_size):
        batches.append(questions[i : i + batch_size])
    return batches
