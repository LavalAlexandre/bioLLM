import litellm 

from openai import OpenAI

#get VLLM_ADDRESS from .env
import os

from dotenv import load_dotenv
load_dotenv()


def get_model_name():
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
    return model_name


def get_model_response(messages, temperature=1, max_tokens=512):
    model=get_model_name()
    api_base = os.getenv("VLLM_ADDRESS")

    response = litellm.completion(
                model=model,
                messages=messages,
                api_base=api_base,
                temperature=temperature,
                max_tokens=max_tokens)
    
    return response
