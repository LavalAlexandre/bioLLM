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


class CompoundSearchTool(Tool):
    """Tool takes a compound name and returns a list of genes associated with that compound from DSigDB."""
    
    name = "search_compound_genes"
    description = "Extracts names of all genes belonging to a given compound from the DSigDB_All.txt dataset using an internal map."
    inputs = {
        "compound_name": {
            "type": "string",
            "description": "The name of the compound to search for (e.g., 'Aspirin' or 'Phenytoin'). Case-insensitive, supports partial matching."
        }
    }
    output_type = "string" # Returns a comma-separated string of gene names

    DSIGDB_FILE_URL = "https://dsigdb.tanlab.org/Downloads/DSigDB_All.txt"

    def __init__(self, tool_name: str = "search_compound_genes"):
        """
        Initializes the CompoundSearchTool and builds the internal compound-to-gene map.
        The map is loaded once upon tool instantiation.
        """
        # Assuming your Tool base class expects a name, passing it here
        super().__init__(tool_name) 
        print(f"Initializing {self.name} and building compound map from {self.DSIGDB_FILE_URL}...")
        try:
            self.map = _build_compound_map_from_url(self.DSIGDB_FILE_URL)
            print("Compound map built successfully.")
        except (ConnectionError, ValueError) as e:
            print(f"Failed to build compound map during initialization: {e}")
            self.map = {} # Initialize an empty map to prevent further errors
        except Exception as e:
            print(f"An unexpected error occurred during map building: {e}")
            self.map = {}


    def _normalize_query_name(self, name: str) -> str:
        """
        Normalizes compound query names for matching against the internal map.
        This is a separate helper for the forward method.
        """
        if not name:
            return ""
        name = name.lower()
        # Remove anything in parentheses (e.g., (CHEBI:XXXX), (CAS:XXXX))
        name = re.sub(r'\s*\(.*\)\s*$', '', name).strip()
        # Remove specific prefixes if they appear at the beginning of the name
        name = re.sub(r'^(chebi|cas|drugbank|pubchem|mesh|unii|hmdb):?\s*', '', name).strip()
        return name

    def forward(self, compound_name: str) -> str:
        """
        Searches the internal DSigDB compound map for the given compound and returns a comma-separated 
        string of associated gene names.
        Returns an informative message if the compound is not found or no genes are associated.
        """
        if not self.map:
            return "Error: Compound map is not loaded. Tool might have failed to initialize properly."

        try:
            # Normalize the search query
            normalized_query = self._normalize_query_name(compound_name)

            if not normalized_query:
                return "Error: Compound name query cannot be empty."

            found_genes = set()
            # Iterate through the keys of the pre-built map for partial matching
            for stored_compound_name, genes in self.map.items():
                if normalized_query in stored_compound_name:
                    found_genes.update(genes)
            
            if not found_genes:
                return f"No genes found for compound matching '{compound_name}' in DSigDB. Try a more general name or check spelling."
            
            return ", ".join(sorted(list(found_genes))) # Return sorted unique genes
        except Exception as e:
            return f"An unexpected error occurred during search: {e}"