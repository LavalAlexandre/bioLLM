import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any
from openai import OpenAI
from transformers import AutoTokenizer
from tqdm import tqdm


def load_questions(filename) -> List[Dict[str, Any]]:
    """Load questions from a JSON file."""
    with open(filename, 'r') as f:
        if filename.endswith('.jsonl'):
            questions = [json.loads(line.strip()) for line in f if line.strip()]
        else:
            questions = json.load(f)
    
    print(f"Loaded {len(questions)} questions from {filename}")
    return questions


# Cache for tokenizers to avoid reloading them
_tokenizer_cache = {}

def get_tokenizer(model: str) -> AutoTokenizer:
    if model not in _tokenizer_cache:
        _tokenizer_cache[model] = AutoTokenizer.from_pretrained(model)
    return _tokenizer_cache[model]


def create_prompts(questions: List[Dict[str, Any]], model_name:str) -> List[str]:
    """
    Create prompts for a batch of questions using AutoTokenizer.
    
    Args:
        questions: List of question data
    
    Returns:
        List of formatted prompts
    """
    tokenizer = get_tokenizer(model_name)
    prompts = []
    
    for question_data in questions:
        question = question_data['question']
        options = question_data.get('options', {})
        
        # Parse options if they're a string
        if isinstance(options, str):
            try:
                options = json.loads(options)
            except:
                options = {}
        
        # Create the prompt with dynamic options
        options_text = ""
        for key, value in options.items():
            options_text += f"{key}: {value}\n"
        
        # Determine the valid options for the system message
        valid_options = list(options.keys())
        options_list = ", ".join(valid_options)
        
        messages = [
            {"role": "system", "content": f"You are a biology expert. Answer the following multiple choice questions by selecting the correct option ({options_list}) and providing a brief explanation. Always format your answer as <answer>[letter]</answer>."},
            {"role": "user", "content": f"""Question: {question}

Options:
{options_text}
Please provide your answer as a single letter ({options_list}) followed by a brief explanation.
Format your answer as: <answer>[letter]</answer>

Answer:"""}
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
        )
        
        prompts.append(text)
    
    return prompts

def make_batches(questions: List[Dict[str, Any]], batch_size: int = 8) -> List[List[Dict[str, Any]]]:
    """
    Split questions into batches for processing.
    
    Args:
        questions: List of question data
        batch_size: Number of questions per batch
    
    Returns:
        List of batches, each containing a list of questions
    """
    batches = []
    for i in range(0, len(questions), batch_size):
        batch = questions[i:i + batch_size]
        batches.append(batch)
    return batches