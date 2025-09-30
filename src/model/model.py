from openai import AsyncOpenAI, OpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel
import os
from dotenv import load_dotenv
import asyncio
from typing import List
from src.model.biorxiv_tool import BiorxivSearchTool
from src.model.cbioportal_tool import CbioportalSearchTool

load_dotenv()

# Ensure OPENAI_API_KEY is set for the agents library
if not os.getenv("OPENAI_API_KEY"):
    print("WARNING: OPENAI_API_KEY not found in .env file")
else:
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

class Model:

    def __init__(self,temperature:int=1, enable_biorxiv:bool=True, enable_cbioportal:bool=True):
        print("Initializing Model and connecting to vLLM server...")
        self.client = OpenAI(
            api_key="EMPTY",  # vLLM doesn't require authentication
            base_url="http://localhost:8000/v1", #vLLM server URL, make sure you have the correct port
        )

        # Create async client for agent
        self.async_client = AsyncOpenAI(
            api_key="EMPTY",
            base_url="http://localhost:8000/v1",
        )

        self.temperature = temperature

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


        # Prepare tools list
        instructions = "You are a multiple choice question answering assistant expert."
        tools = []
        tool_descriptions = []
        
        if enable_biorxiv:
            tools.append(BiorxivSearchTool)
            tool_descriptions.append("bioRxiv research papers")
            print("✓ BioRxiv search tool enabled")
        
        if enable_cbioportal:
            tools.append(CbioportalSearchTool)
            tool_descriptions.append("cBioPortal mutation data")
            print("✓ cBioPortal search tool enabled")
        
        # Build instructions based on enabled tools
        if tools:
            tool_list = " and ".join(tool_descriptions)
            instructions = f"""You are a biology expert assistant with access to {tool_list}.

            When answering questions:
            1. Identify key biological entities in the question (genes, proteins, cancer types, diseases, biological processes).
            2. Use appropriate tools to gather relevant information:
            - Use search_biorxiv for recent research, literature, and general biological knowledge
            - Use search_cbioportal for gene mutation statistics in specific cancer types
            3. Review the search results carefully to inform your answer.
            4. After gathering information, provide your answer in the format: <answer>[letter]</answer>

            For multiple choice questions, select the correct option and provide a brief explanation based on the research findings."""
        else:
            instructions = "You are a concise multiple choice question answering assistant expert."

        # Define the agent with async client
        self.agent = Agent(
            name="Openai agent",
            instructions=instructions,
            model=OpenAIChatCompletionsModel(
                model=self.model_name,
                openai_client=self.async_client,
            ),
        tools=tools if tools else None,
        )


    def completion(self, prompts, temperature=1, max_tokens=512):
        """
        Generate completions for a single prompt or batch of prompts.
        
        Args:
            prompts: Either a single string prompt or a list of string prompts
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            A response object with choices attribute
        """
        # Use the completions API for batch processing
        response = self.client.completions.create(
            model=self.model_name,
            prompt=prompts,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return response

    async def agent_completion(self, input_text: str):
        """
        Generate completions using the agent framework for a single prompt.
        
        Args:
            input_text: The input prompt/question
            
        Returns:
            The agent's response
        """
        result = await Runner.run(self.agent, input=input_text)
        return result

    async def agent_batch_completion(self, inputs: List[str], max_concurrent: int = 10):
        """
        Generate completions using the agent framework for multiple prompts concurrently.
        
        Args:
            inputs: List of input prompts/questions
            max_concurrent: Maximum number of concurrent requests (default: 10)
            
        Returns:
            List of agent responses in the same order as inputs
        """
        # Create a semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(input_text: str):
            async with semaphore:
                return await self.agent_completion(input_text)
        
        # Process all inputs concurrently with semaphore limit
        tasks = [process_with_semaphore(input_text) for input_text in inputs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return results