import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any
from openai import OpenAI
from transformers import AutoTokenizer
from tqdm import tqdm

from src.data_preprocess import load_questions,create_prompts,make_batches

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#7Configuration for batch processing
BATCH_SIZE = 256  # Number of questions per batch. Increasing this will speed up the processing at the cost of using more memory
MAX_TOKENS_PER_QUESTION = 2_000  # Max tokens per individual question
TEMPERATURE = 0.7  # Model temp

# Initialize the OpenAI client to connect to vLLM server
client = OpenAI(
    api_key="EMPTY",  # vLLM doesn't require authentication
    base_url="http://localhost:8000/v1", #vLLM server URL, make sure you have the correct port
)

# Test the connection
try:
    models = list(client.models.list())
    if models:
        print("vLLM server is up and running!")
        print(f"Available models: {[model.id for model in models]}")
        model_name = models[0].id
    else:
        raise Exception("No models available")
except Exception as e:
    print(f"Error connecting to vLLM server: {e}")
    print("Make sure the server is running by executing `./start_vllm_docker.sh`")
    raise e 


def extract_answer_from_response(response_text: str, question_data: Dict[str, Any]) -> str:
    """
    Extract the answer letter (A, B, C, D, etc.) from the model response.
    
    Args:
        response_text: The raw response from the model
        question_data: The question data containing options
    
    Returns:
        The answer letter (A, B, C, D, etc.) or 'X' if not found
    """
    # Clean the response text
    response_text = response_text.strip().upper()
    
    # Get available options from the question data
    options = question_data.get('options', {})
    if isinstance(options, str):
        try:
            options = json.loads(options)
        except:
            options = {}
    
    # Extract valid option letters (A, B, C, D, etc.)
    valid_options = list(options.keys()) if isinstance(options, dict) else ['A', 'B', 'C', 'D', 'E']
    valid_pattern = '|'.join(valid_options)
    
    # Look for patterns like "A", "B", "Answer: A", "The answer is B", etc.
    patterns = [
        rf'<answer>([{valid_pattern}])</answer>',  # Look for the required format first, then fallback patterns
        rf'[aA]nswer[\s:]*([{valid_pattern}])',
        rf'\b([{valid_pattern}])\b',
        rf'option[\s:]*([{valid_pattern}])',
        rf'choice[\s:]*([{valid_pattern}])',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    # If no clear answer found, try to match against the actual options
    if isinstance(options, dict):
        for key, value in options.items():
            if value.lower() in response_text.lower():
                return key.upper()
    
    # Default to 'X' if no answer found
    return 'X'



def generate_completions(questions, model_name, output_filename="answers.jsonl") -> List[Dict[str, Any]]:
    """Generate completions for a list of questions."""
    batches = make_batches(questions, BATCH_SIZE)
    print(f"Processing {len(questions)} questions in {len(batches)} batches...")
    
    results = []
    for batch_idx, batch in tqdm(enumerate(batches), total=len(batches)):
        print(f"\nProcessing batch {batch_idx + 1}/{len(batches)} ({len(batch)} questions)...")
        
        try:
            prompts = create_prompts(batch, model_name)
            response = client.completions.create(
                model=model_name,
                prompt=prompts,
                max_tokens=MAX_TOKENS_PER_QUESTION,
                temperature=TEMPERATURE
            )
            
            for i, (question_data, choice) in enumerate(zip(batch, response.choices)):
                response_text = choice.text
                answer_letter = extract_answer_from_response(response_text, question_data)
                
                result = {
                    **question_data,
                    'raw_response': response_text,
                    'answer_letter': answer_letter
                }
                results.append(result)
                
        except Exception as e:
            print(f"Error processing batch {batch_idx + 1}: {e}")
            for question_data in batch:
                result = {
                    **question_data,
                    'raw_response': f'Error: {str(e)}',
                    'answer_letter': 'X'
                }
                results.append(result)
    
    # Save results
    with open(output_filename, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"Results saved to {output_filename}")
    return results

# Process test questions


def classify_file(file, model):
    if Path(file).exists():
        questions = load_questions(file)
        results = generate_completions(questions, model_name, "result/test_answers.jsonl")
        print("✅ Test questions processed!")
    else:
        print(f"ℹ️  No {file} found in directory")
    
