from openai import AsyncOpenAI, OpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel, ModelSettings
import os
from dotenv import load_dotenv
import asyncio
import json
import re
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

        # Initialize specialized agents
        self._init_agents(enable_biorxiv=enable_biorxiv, enable_cbioportal=enable_cbioportal)
    def _init_agents(self, enable_biorxiv: bool, enable_cbioportal: bool):
        """Initialize the multi-agent system with different token budgets"""
        
        # Build tool list first - needed for search agent
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
        
        tool_list = " and ".join(tool_descriptions) if tools else "no external tools"
        
        # 1. Planning Agent - lightweight, focused on entity extraction (NO TOOLS)
        self.planning_agent = Agent(
            name="Planning Agent",
            instructions="""You are a planning agent. Extract key information and output JSON immediately.

            Your JSON must have this exact structure:
            {
                "entities": ["gene1", "gene2", "cancer_type"],
                "tools": ["cbioportal"],
                "queries": ["query1", "query2"]
            }
            
            Rules:
            - Extract gene names from the question
            - Extract cancer type from the question
            - Use "cbioportal" as the tool for gene/cancer queries
            - Create 1-2 specific search queries
            
            Output the JSON directly. No explanation needed.""",
            model=OpenAIChatCompletionsModel(
                model=self.model_name,
                openai_client=self.async_client,
            ),
            model_settings=ModelSettings(
                max_tokens=800,  # 
                temperature=0.1,  # Very low temperature for focused output
                frequency_penalty=0.7,  # Strong penalty against repetition
            ),
            tools=[],
        )

        # 2. Search Agent - executes searches based on plan (HAS TOOLS)
        self.search_agent = Agent(
            name="Search Agent",
            instructions=f"""You are a search execution agent with access to {tool_list}.
            
            You receive a JSON plan. Execute the searches using the tools.
            
            Output your findings as JSON:
            {{
                "results": [
                    {{"source": "cbioportal", "data": "mutation data here"}},
                ]
            }}
            
            Execute tools and report findings concisely.""",
            model=OpenAIChatCompletionsModel(
                model=self.model_name,
                openai_client=self.async_client,
            ),
            model_settings=ModelSettings(
                max_tokens=2000,  # More tokens for tool execution
                temperature=0.3,
            
            ),
            tools=tools,
        )

        # 3. Conclusion Agent - deep reasoning with gathered data (NO TOOLS)
        self.conclusion_agent = Agent(
            name="Conclusion Agent",
            instructions="""You are an expert biological reasoning agent.

            You receive:
            - Original question with answer options
            - Search results (JSON)
            
            Think deeply about:
            - What the search results tell you
            - Which answer option is supported by the evidence
            - Biological mechanisms involved
            
            After reasoning, provide your answer: <answer>[letter]</answer>
            
            You have 2048 tokens - use them to think thoroughly.""",
            model=OpenAIChatCompletionsModel(
                model=self.model_name,
                openai_client=self.async_client,
            ),
            model_settings=ModelSettings(
                max_tokens=3048,
                temperature=self.temperature,
                frequency_penalty=0.0,
                presence_penalty=0.2,
            ),
            tools=[],
        )

    def _extract_json(self, text: str) -> str:
        """Extract JSON from agent output, handling thinking blocks"""
        # Try to find JSON in the text
        # Look for {...} pattern
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if json_match:
            try:
                # Validate it's proper JSON
                json.loads(json_match.group())
                return json_match.group()
            except json.JSONDecodeError:
                pass
        
        # If no valid JSON found, return the original text
        return text

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
        Generate completions using the multi-agent framework.
        
        Args:
            input_text: The input prompt/question
            
        Returns:
            The final agent's response
        """
        # Extract just the question text (without options) for planning/search
        base_question = self._extract_base_question(input_text)
        
        # Step 1: Planning (limited tokens - 300 max)
        plan_result = await Runner.run(
            self.planning_agent, 
            input=f"Create a search plan for this question:\n{base_question}"
        )

        
        # Extract JSON from plan result
        plan_text = plan_result.text if hasattr(plan_result, 'text') else str(plan_result)
        plan_json = self._extract_json(plan_text)
        
        # Step 2: Search execution (moderate tokens - 800 max)
        # Only provide the base question, not the answer options
        search_input = f"""Search plan:
{plan_json}

Execute searches for the question:
{base_question}"""
        
        search_result = await Runner.run(
            self.search_agent, 
            input=search_input,
        )
        
        # Extract JSON from search results
        search_text = search_result.text if hasattr(search_result, 'text') else str(search_result)
        search_json = self._extract_json(search_text)
        
        # Step 3: Conclusion (most tokens - 2048 max for deep reasoning)
        # NOW provide the full question with options for final reasoning
        conclusion_input = f"""Question:
{input_text}

Search Plan:
{plan_json}

Search Results:
{search_json}

Analyze and provide your final answer."""
        
        final_result = await Runner.run(
            self.conclusion_agent, 
            input=conclusion_input,
        )
        
        return final_result


    def _extract_base_question(self, input_text: str) -> str:
        """Extract just the question text without answer options and instructions"""
        # Remove everything from "Options:" onward
        if "Options:" in input_text:
            parts = input_text.split("Options:", 1)
            question_part = parts[0].strip()
        else:
            question_part = input_text
        
        # Remove everything from "Please provide" onward (catches both variants)
        if "Please provide" in question_part:
            question_part = question_part.split("Please provide", 1)[0].strip()
        
        # Remove "Question:" prefix if present
        if question_part.startswith("Question:"):
            question_part = question_part.replace("Question:", "", 1).strip()
        
        # Remove "Format your answer" if somehow still present
        if "Format your answer" in question_part:
            question_part = question_part.split("Format your answer", 1)[0].strip()
        
        return question_part


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