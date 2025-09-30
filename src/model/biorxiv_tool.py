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


class BiorxivSearchTool(Tool):
    """Tool for searching arXiv using DuckDuckGo."""
    
    name = "search_biorxiv"
    description = "Search biorxiv for preprints related to biology topics"
    inputs = {
        "query": {
            "type": "string",
            "description": "The search query for biorxiv preprints"
        }
    }
    output_type = "string"
    
    def forward(self, query: str) -> str:
        """Search arXiv using DuckDuckGo site search."""
        search_query = f"site:biorxiv.org {query}"
        return duckduckgo_tool(search_query)