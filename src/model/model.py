from openai import AsyncOpenAI, OpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel, ModelSettings
from dotenv import load_dotenv
import asyncio
from typing import List, Optional
import httpx
from src.model.biorxiv_tool import BiorxivSearchTool
from src.model.cbioportal_tool import search_cbioportal
from src.model.protein_expression_tool import ProteinExpressionTool
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()


class Model:
    def __init__(
        self,
        temperature: float = 0.6,
        enable_biorxiv: bool = False,
        enable_cbioportal: bool = True,
        enable_protein_expression: bool = True,
        max_concurrent: Optional[int] = None,
        request_timeout: float = 400.0,
    ):
        logger.info("Initializing Model and connecting to vLLM server...")

        # Connection pooling for better performance
        http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(request_timeout, connect=10.0),
            limits=httpx.Limits(
                max_keepalive_connections=100,
                max_connections=150,
                keepalive_expiry=30.0,
            ),
        )

        self.client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
        self.async_client = AsyncOpenAI(
            api_key="EMPTY",
            base_url="http://localhost:8000/v1",
            http_client=http_client,
        )
        self.temperature = temperature
        self.request_timeout = request_timeout

        models = list(self.client.models.list())
        if not models:
            raise Exception("No models available on vLLM server")

        self.model_name = models[0].id
        logger.info(f"Using model: {self.model_name}")

        # Dynamic concurrency based on GPU count
        # vLLM with 8 GPUs can handle 64-96 concurrent requests efficiently
        # Rule of thumb: 4-8x concurrent requests per GPU for inference-only
        # With tool calls (network I/O latency): 8-12x per GPU to hide I/O wait time
        if max_concurrent is None:
            # Auto-detect: 8 GPUs with tools (cBioPortal, protein expression APIs)
            # For tensor-parallel-size=8, optimal is 64-96 concurrent
            # Starting with 80 (10x per GPU) to balance GPU utilization and tool latency
            max_concurrent = 80
            logger.info(
                f"Auto-configured concurrency: {max_concurrent} (optimized for 8 GPUs + tool latency)"
            )
        else:
            logger.info(f"Using custom concurrency: {max_concurrent}")

        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.max_concurrent = max_concurrent

        self._init_agent(enable_biorxiv, enable_cbioportal, enable_protein_expression)

    def _init_agent(
        self,
        enable_biorxiv: bool,
        enable_cbioportal: bool,
        enable_protein_expression: bool,
    ):
        """Initialize single agent with optional tools"""
        tools = []

        if enable_biorxiv:
            tools.append(BiorxivSearchTool)
            logger.info("✓ BioRxiv search enabled")

        if enable_cbioportal:
            tools.append(search_cbioportal)
            logger.info("✓ cBioPortal search enabled")

        if enable_protein_expression:
            tools.append(ProteinExpressionTool)
            logger.info("✓ Protein expression search enabled")

        self.agent = Agent(
            name="Bio Agent",
            instructions="""Expert biological reasoning agent. Answer questions efficiently.

CRITICAL WORKFLOW:
1. If question involves genes/proteins/cancer data → IMMEDIATELY use tools (cBioPortal/protein expression)
2. Use tool results to answer → keep reasoning minimal
3. Format final answer as: <answer>[letter]</answer>

DO NOT:
- Overthink before calling tools
- Provide lengthy explanations before tool calls
- Repeat information already in tool results

DO:
- Call tools immediately for data-dependent questions
- Be concise and direct
- Trust tool data over general knowledge""",
            model=OpenAIChatCompletionsModel(
                model=self.model_name,
                openai_client=self.async_client,
            ),
            model_settings=ModelSettings(
                max_tokens=4096,  # Increased for complex tool calls (prevents truncation)
                # Qwen-specific settings (from official docs):
                # - DO NOT use greedy decoding (temperature=0) - causes performance degradation
                # - Use presence_penalty 0-2 to reduce endless repetitions
                # - Higher values may cause language mixing but prevent loops
                temperature=0.6,  # Balanced speed/quality, avoids greedy decoding
                top_p=0.95,  # Nucleus sampling for quality
                top_k=20,  # Limits vocabulary for faster generation
                frequency_penalty=0.0,  # Not needed with presence_penalty
                presence_penalty=1.0,  # Qwen recommendation: 0-2 range to prevent repetition
            ),
            tools=tools,
        )

    def completion(self, prompts, temperature=0.6, max_tokens=512):
        """Generate completions for single prompt or batch"""
        # Ensure temperature > 0 (Qwen requirement: no greedy decoding)
        if temperature == 0.0:
            logger.warning(
                "Temperature=0 (greedy decoding) not recommended for Qwen. Using 0.3 instead."
            )
            temperature = 0.3

        return self.client.completions.create(
            model=self.model_name,
            prompt=prompts,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95,
            top_k=20,
            presence_penalty=1.0,  # Qwen: reduce repetitions
        )

    async def agent_completion(self, input_text: str):
        """Generate completion using single agent with semaphore control"""
        async with self.semaphore:
            try:
                # Add timeout to prevent hanging requests
                result = await asyncio.wait_for(
                    Runner.run(self.agent, input=input_text),
                    timeout=self.request_timeout,
                )
                return result
            except asyncio.TimeoutError:
                logger.error(f"Request timeout after {self.request_timeout}s")
                raise TimeoutError(f"Request exceeded {self.request_timeout}s timeout")
            except Exception as e:
                logger.error(f"Error in agent_completion: {e}")
                raise

    async def agent_batch_completion(
        self, inputs: List[str], max_concurrent: Optional[int] = None
    ):
        """
        Generate completions for multiple prompts concurrently.
        Optimized for vLLM's continuous batching engine.

        Args:
            inputs: List of input texts
            max_concurrent: Override the default semaphore limit (None = use model default)

        Returns:
            List of results (or exceptions if return_exceptions=True)
        """
        logger.info(f"Starting batch of {len(inputs)} requests")

        if max_concurrent:
            # Temporarily override semaphore for this batch
            semaphore = asyncio.Semaphore(max_concurrent)
            logger.info(
                f"Using custom concurrency: {max_concurrent} for batch of {len(inputs)}"
            )

            async def process_with_limit(idx: int, input_text: str):
                async with semaphore:
                    try:
                        logger.debug(f"Processing request {idx + 1}/{len(inputs)}")
                        result = await asyncio.wait_for(
                            Runner.run(self.agent, input=input_text),
                            timeout=self.request_timeout,
                        )
                        logger.debug(f"Completed request {idx + 1}/{len(inputs)}")
                        return result
                    except asyncio.TimeoutError:
                        logger.warning(
                            f"Request {idx} timed out after {self.request_timeout}s"
                        )
                        raise TimeoutError(
                            f"Request exceeded {self.request_timeout}s timeout"
                        )
                    except Exception as e:
                        logger.warning(f"Request {idx} failed: {e}")
                        raise
        else:
            # Use the model's default semaphore (optimized for GPU count)
            async def process_with_limit(idx: int, input_text: str):
                logger.debug(f"Processing request {idx + 1}/{len(inputs)}")
                result = await self.agent_completion(input_text)
                logger.debug(f"Completed request {idx + 1}/{len(inputs)}")
                return result

        # Create all tasks at once - vLLM's continuous batching will optimize scheduling
        tasks = [process_with_limit(idx, text) for idx, text in enumerate(inputs)]

        logger.info(f"Created {len(tasks)} tasks, gathering results...")

        # Gather all results, keeping exceptions for error handling
        results = await asyncio.gather(*tasks, return_exceptions=True)

        logger.info(f"Batch complete: {len(results)} results received")
        return results
