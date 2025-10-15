from pathlib import Path
from typing import List, Dict, Any

from tqdm.asyncio import tqdm
from src.model.model import Model
from src.data_preprocess import load_questions, create_prompts, make_batches
import json
import re
import logging
import asyncio
import gc

TESTING_CONFIG = False


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# OPTIMIZATION CONFIGURATION
# ============================================================================

# Batch size: Number of questions to process per batch
# Recommendations:
#   - Small (8-16): Low latency, real-time processing
#   - Medium (32-64): Balanced throughput and latency
#   - Large (128-256): Maximum throughput
# With 8 GPUs: 64-128 is optimal
BATCH_SIZE = 32  # Optimized for 8-GPU setup

if TESTING_CONFIG:
    BATCH_SIZE = 1

# Max tokens per question
# Lower = faster generation
# Most biology questions need 1024-2048 tokens
MAX_TOKENS_PER_QUESTION = 2048

# Model temperature (0.0-1.0)
# Lower = faster (less sampling), more deterministic
TEMPERATURE = 0.6  # Balanced

# ============================================================================


def extract_answer_from_response(
    response_text: str, question_data: Dict[str, Any]
) -> str:
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
    options = question_data.get("options", {})
    if isinstance(options, str):
        try:
            options = json.loads(options)
        except (json.JSONDecodeError, ValueError):
            options = {}

    # Extract valid option letters (A, B, C, D, etc.)
    valid_options = (
        list(options.keys()) if isinstance(options, dict) else ["A", "B", "C", "D", "E"]
    )
    valid_pattern = "|".join(valid_options)

    # Look for patterns like "A", "B", "Answer: A", "The answer is B", etc.
    patterns = [
        rf"<answer>([{valid_pattern}])</answer>",
        rf"[aA]nswer[\s:]*([{valid_pattern}])",
        rf"\b([{valid_pattern}])\b",
        rf"option[\s:]*([{valid_pattern}])",
        rf"choice[\s:]*([{valid_pattern}])",
    ]

    for pattern in patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    # If no clear answer found, try to match against the actual options
    if isinstance(options, dict):
        for key, value in options.items():
            if value.lower() in response_text.lower():
                return key

    # Default to 'X' if no answer found
    return "X"


async def generate_completions_with_agent(
    questions: List[Dict[str, Any]],
    model: Model,
    output_filename: str = "answers.jsonl",
) -> List[Dict[str, Any]]:
    """Generate completions using agent with simple batch processing."""
    print(f"\n{'=' * 70}")
    print(f"Processing {len(questions)} questions using agent")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"{'=' * 70}\n")

    # Process in batches with progress tracking
    batches = make_batches(questions, BATCH_SIZE)
    all_results = []

    print(f"Total batches to process: {len(batches)}")

    # Open output file for incremental writing (memory efficient)
    with open(output_filename, "w") as f:
        for batch_idx, batch in enumerate(batches):
            print(f"\n{'=' * 50}")
            print(
                f"[Batch {batch_idx + 1}/{len(batches)}] Processing {len(batch)} questions..."
            )
            print(f"{'=' * 50}")

            # Get prompts for this batch - use_agent=True for clean question text
            batch_prompts = create_prompts(batch, model.model_name, use_agent=True)
            print(f"Created {len(batch_prompts)} prompts")

            try:
                print("Sending batch to vLLM...")
                import time

                start_time = time.time()

                # Run batch completion concurrently
                batch_responses = await model.agent_batch_completion(batch_prompts)

                elapsed = time.time() - start_time
                print(f"Received {len(batch_responses)} responses in {elapsed:.1f}s")
                print(f"Average: {elapsed / len(batch_responses):.2f}s per question")

                # Process responses
                success_count = 0
                error_count = 0

                for question_data, result_obj in zip(batch, batch_responses):
                    # Handle exceptions from gather
                    if isinstance(result_obj, Exception):
                        error_count += 1
                        logger.warning(
                            f"Error processing question {question_data.get('id', 'unknown')}: {result_obj}"
                        )
                        response_text = f"Error: {str(result_obj)}"
                        answer_letter = "X"
                    else:
                        success_count += 1
                        # Extract response text from agent result
                        response_text = (
                            str(result_obj.final_response)
                            if hasattr(result_obj, "final_response")
                            else str(result_obj)
                        )
                        answer_letter = extract_answer_from_response(
                            response_text, question_data
                        )

                    result = {
                        **question_data,
                        "raw_response": response_text,
                        "answer_letter": answer_letter,
                    }
                    all_results.append(result)

                    # Write result incrementally (memory efficient, crash-safe)
                    f.write(json.dumps(result) + "\n")
                    f.flush()  # Ensure data is written to disk

                print(
                    f"[Batch {batch_idx + 1}/{len(batches)}] Success: {success_count}, Errors: {error_count}"
                )

                # Hint to garbage collector (helps with memory)
                if batch_idx % 10 == 0:
                    gc.collect()

            except Exception as e:
                logger.error(f"Fatal error processing batch {batch_idx + 1}: {e}")
                # Still save error results
                for question_data in batch:
                    result = {
                        **question_data,
                        "raw_response": f"Error: {str(e)}",
                        "answer_letter": "X",
                    }
                    all_results.append(result)
                    f.write(json.dumps(result) + "\n")
                    f.flush()

    print(f"\n{'=' * 70}")
    print(f"Results saved to {output_filename}")
    print(f"Total processed: {len(all_results)}")
    print(f"{'=' * 70}\n")

    return all_results


def generate_completions(
    questions: List[Dict[str, Any]],
    model: Model,
    output_filename: str = "answers.jsonl",
) -> List[Dict[str, Any]]:
    """Generate completions for a list of questions (legacy batch mode)."""

    if TESTING_CONFIG:
        BATCH_SIZE = 1

    batches = make_batches(questions, BATCH_SIZE)
    if TESTING_CONFIG:
        batches = batches[:1]  # Only process first batch for testing

    print(f"Processing {len(questions)} questions in {len(batches)} batches...")

    results = []
    for batch_idx, batch in tqdm(enumerate(batches), total=len(batches)):
        print(
            f"\nProcessing batch {batch_idx + 1}/{len(batches)} ({len(batch)} questions)..."
        )

        try:
            # use_agent=False for tokenizer chat template
            prompts = create_prompts(batch, model.model_name, use_agent=False)
            response = model.completion(
                prompts=prompts,
                max_tokens=MAX_TOKENS_PER_QUESTION,
                temperature=TEMPERATURE,
            )

            for i, (question_data, choice) in enumerate(zip(batch, response.choices)):
                response_text = choice.text
                answer_letter = extract_answer_from_response(
                    response_text, question_data
                )

                result = {
                    **question_data,
                    "raw_response": response_text,
                    "answer_letter": answer_letter,
                }
                results.append(result)

        except Exception as e:
            print(f"Error processing batch {batch_idx + 1}: {e}")
            for question_data in batch:
                result = {
                    **question_data,
                    "raw_response": f"Error: {str(e)}",
                    "answer_letter": "X",
                }
                results.append(result)

    # Save results
    with open(output_filename, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    print(f"Results saved to {output_filename}")
    return results


async def classify_file_async(file, model, use_agent=True):
    """Async version of classify_file that supports agent completion."""
    if Path(file).exists():
        print(f"ℹ️  Found {file} in directory, processing test questions...")
        questions = load_questions(file)
        print(f"ℹ️  Loaded {len(questions)} questions from {file}")
        print("ℹ️  Generating completions...")

        if use_agent:
            await generate_completions_with_agent(
                questions, model, "result/test_answers.jsonl"
            )
        else:
            generate_completions(questions, model, "result/test_answers.jsonl")

        print("✅ Test questions processed!")
    else:
        print(f"ℹ️  No {file} found in directory")


def classify_file(file, model, use_agent=True):
    """Wrapper to run async classify_file."""
    asyncio.run(classify_file_async(file, model, use_agent))
