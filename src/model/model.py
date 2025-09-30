from openai import OpenAI
from types import SimpleNamespace
import os
from dotenv import load_dotenv
load_dotenv()


class Model:

    def __init__(self):
        print("Initializing Model and connecting to vLLM server...")
        self.client = OpenAI(
            api_key="EMPTY",  # vLLM doesn't require authentication
            base_url="http://localhost:8000/v1", #vLLM server URL, make sure you have the correct port
        )

        # Test the connection
        try:
            models = list(self.client.models.list())
            if models:
                print("vLLM server is up and running!")
                print(f"Available models: {[model.id for model in models]}")
                model_name = models[0].id
                self.model_name = model_name
            else:
                raise Exception("No models available")
        except Exception as e:
            print(f"Error connecting to vLLM server: {e}")
            print("Make sure the server is running by executing `./start_vllm_docker.sh`")
            raise e 
        print(f"Using model: {self.model_name}")



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