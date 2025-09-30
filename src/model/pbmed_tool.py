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

duckduckgo_tool = DuckDuckGoSearchTool()

class PubMedSearchTool(Tool):
    """Tool for searching PubMed using DuckDuckGo."""
    
    name = "search_pubmed"
    description = "Search PubMed for scientific articles related to biology topics"
    inputs = {
        "query": {
            "type": "string",
            "description": "The search query for PubMed articles"
        }
    }
    output_type = "string"
    
    def forward(self, query: str) -> str:
        """Search PubMed using DuckDuckGo site search."""
        search_query = f"site:pubmed.ncbi.nlm.nih.gov {query}"
        return duckduckgo_tool(search_query)
