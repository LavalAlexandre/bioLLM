from typing import List, Tuple, Dict, Optional, Any
import requests
from collections import Counter
from typing import List, Tuple, Dict, Optional, Any
from smolagents import DuckDuckGoSearchTool
import json
import logging
from openai import OpenAI
from smolagents import CodeAgent, OpenAIServerModel, Tool

import requests
from io import StringIO
import pandas as pd
from collections import defaultdict
import re


from src.model.pbmed_tool import PubMedSearchTool
from src.model.compound_tool import CompoundSearchTool
from src.model.cbioportal_tool import cbioportal_tool
from src.model.biorxiv_tool import BiorxivSearchTool


duckduckgo_tool = DuckDuckGoSearchTool()
# Initialize the OpenAI client to connect to vLLM server
client = OpenAI(
    api_key="EMPTY",  # vLLM doesn't require authentication
    base_url="http://localhost:8000/v1",
)
# Retrieve the model id from the vLLM server
try:
    model_ids = list(client.models.list())
    if model_ids:
        print("vLLM server is up and running!")
        print(f"Available models: {[model.id for model in model_ids]}")
    else:
        print("No models available")
except Exception as e:
    print(f"Error connecting to vLLM server: {e}")
    print("Make sure the server is running by executing `./start_vllm_docker.sh`")
model_id = model_ids[0].id


# Initialize the smolagents model to connect to our vLLM server
model = OpenAIServerModel(
    model_id=model_id,
    api_base="http://localhost:8000/v1",
    api_key="EMPTY",
    temperature=0.7
)


def create_code_agent(model,verbose=False):
    # Create the custom tools
    pubmed_tool = PubMedSearchTool()
    #biorxiv_tool = BiorxivSearchTool()
    #cbioportal_tool = CbioportalSearchTool()
    #compound_tool = CompoundSearchTool()

    
    # Create agent with multiple search tools
    agent = CodeAgent(
        #tools=[duckduckgo_tool, pubmed_tool, biorxiv_tool, cbioportal_tool, compound_tool],
        tools=[pubmed_tool],
        model=model,
        stream_outputs=verbose,
        logger=None
    )
    return agent