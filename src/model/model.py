import litellm 
from litellm import batch_completion_models_all_responses

from openai import OpenAI
from types import SimpleNamespace

#get VLLM_ADDRESS from .env
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


    def completion(self, messages, temperature=1, max_tokens=512):
        model = self.model_name
        api_base = os.getenv("VLLM_ADDRESS", "http://localhost:8000/v1")
        api_key = os.getenv("VLLM_API_KEY", "EMPTY")
        
        # Normalize messages to list of message lists
        if isinstance(messages, list) and all(not isinstance(item, dict) for item in messages):
            # List of simple prompts - batch them
            message_batches = [
                [{"role": "user", "content": str(item)}]
                for item in messages
            ]
            
            try:
                responses = batch_completion_models_all_responses(
                    models=[model] * len(message_batches),
                    messages=message_batches,
                    api_base=api_base,
                    api_key=api_key,
                    custom_llm_provider="openai",
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                print(f"DEBUG: Batch responses type: {type(responses)}")
                print(f"DEBUG: Batch responses length: {len(responses) if hasattr(responses, '__len__') else 'N/A'}")
                if isinstance(responses, list) and len(responses) > 0:
                    print(f"DEBUG: First response type: {type(responses[0])}")
                    print(f"DEBUG: First response: {responses[0]}")
                
                return [self._wrap_response(resp) for resp in responses]
            except Exception as e:
                print(f"DEBUG: Error in batch_completion: {e}")
                print(f"DEBUG: Error type: {type(e)}")
                import traceback
                traceback.print_exc()
                raise
        
        # Single message or already formatted chat messages
        if isinstance(messages, list):
            chat_messages = [
                item if isinstance(item, dict)
                else {"role": "user", "content": str(item)}
                for item in messages
            ]
        else:
            chat_messages = [{"role": "user", "content": str(messages)}]
        
        raw_response = litellm.completion(
            model=model,
            messages=chat_messages,
            api_base=api_base,
            api_key=api_key,
            custom_llm_provider="openai",
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return self._wrap_response(raw_response)
    
    def _wrap_response(self, raw_response):
        # Debug: print the type and structure
        print(f"DEBUG: raw_response type: {type(raw_response)}")
        if isinstance(raw_response, list) and len(raw_response) > 0:
            print(f"DEBUG: First item type: {type(raw_response[0])}")
            print(f"DEBUG: First item: {raw_response[0]}")
        
        # Handle if raw_response is already a list (from batch operations)
        if isinstance(raw_response, list):
            return [self._wrap_single_response(resp) for resp in raw_response]
        return self._wrap_single_response(raw_response)
    
    def _wrap_single_response(self, raw_response):
        # Handle both dict and object responses
        def safe_get(obj, key, default=None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)
        
        # Handle list of responses (batch_completion_models_all_responses returns list of lists)
        if isinstance(raw_response, list):
            if len(raw_response) > 0:
                raw_response = raw_response[0]  # Get first item if it's a list
            else:
                return SimpleNamespace(choices=[])
        
        def _to_choice(choice_dict):
            text = safe_get(choice_dict, "text")
            if text is None:
                message = safe_get(choice_dict, "message") or {}
                text = safe_get(message, "content", "")
            return SimpleNamespace(
                text=text,
                index=safe_get(choice_dict, "index"),
                logprobs=safe_get(choice_dict, "logprobs"),
                finish_reason=safe_get(choice_dict, "finish_reason")
            )
        
        choices = safe_get(raw_response, "choices", [])
        
        return SimpleNamespace(
            id=safe_get(raw_response, "id"),
            object=safe_get(raw_response, "object"),
            created=safe_get(raw_response, "created"),
            model=safe_get(raw_response, "model"),
            usage=safe_get(raw_response, "usage"),
            choices=[_to_choice(choice) for choice in choices]
        )
