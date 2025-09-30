import litellm 

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
                    max_tokens=max_tokens)

        print(f"Raw response from litellm: {raw_response}")

        def _to_choice(choice_dict):
            text = choice_dict.get("text")
            if text is None:
                message = choice_dict.get("message") or {}
                text = message.get("content", "")
            return SimpleNamespace(
                text=text,
                index=choice_dict.get("index"),
                logprobs=choice_dict.get("logprobs"),
                finish_reason=choice_dict.get("finish_reason")
            )
        
        def _wrap_response(raw_response):
            return SimpleNamespace(
                id=raw_response.get("id"),
                object=raw_response.get("object"),
                created=raw_response.get("created"),
                model=raw_response.get("model"),
                usage=raw_response.get("usage"),
                choices=[_to_choice(choice) for choice in raw_response.get("choices", [])]
            )

        def _dispatch(chat_messages):
            raw_response = litellm.completion(
                        model=model,
                        messages=chat_messages,
                        api_base=api_base,
                        api_key=api_key,
                        custom_llm_provider="openai",
                        temperature=temperature,
                        max_tokens=max_tokens)
            print(f"Raw response from litellm: {raw_response}")
            return _wrap_response(raw_response)

        if isinstance(messages, list) and all(not isinstance(item, dict) for item in messages):
            return [
                _dispatch([{"role": "user", "content": str(item)}])
                for item in messages
            ]

        if isinstance(messages, list):
            chat_messages = [
                item if isinstance(item, dict)
                else {"role": "user", "content": str(item)}
                for item in messages
            ]
        else:
            chat_messages = [{"role": "user", "content": str(messages)}]

        return _dispatch(chat_messages)