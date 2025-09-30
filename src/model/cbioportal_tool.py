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

class CbioportalSearchTool(Tool):
    """
    Tool for searching cBioPortal to get a summary of gene mutations for a specific cancer type.
    """
    name = "cbioportal_gene_mutation_summary"
    description = (
        "Retrieves gene mutation statistics from cBioPortal for a given cancer type. "
        "Use this to find the mutation frequency and a compact mutation profile for one or more genes in a specific cancer."
        """
        Output Description:
        The output is a JSON string representing a dictionary where each key is a gene symbol (str).
        The value for each gene is another dictionary with the following structure:
        {
            "mutation_frequency": float,
            "compact_profile": str
        }
        
        - "mutation_frequency" (float): The fraction of unique patients in the study who have a mutation in this gene.
        - "compact_profile" (str): A pipe-separated string summarizing mutation features in the format: "freq:<val>|avg_vaf:<val>|top_type:<val>"
            - "freq": The mutation frequency, formatted to two decimal places.
            - "avg_vaf": The average Variant Allele Frequency (VAF) across all mutations in the gene, formatted to two decimal places.
            - "top_type": The most common mutation type (e.g., 'Missense_Mutation').
        """
    )
    inputs = {
        "genes": {
            "type": "string",
            "description": "A comma-separated list of gene symbols. For example: 'TP53,PIK3CA'."
        },
        "cancer_name": {
            "type": "string",
            "description": "The name of the cancer study. For example: 'Breast Invasive Carcinoma'."
        }
    }
    output_type = "string"

    def _get_first_study_id(self, keyword: str) -> Optional[str]:
        """Fetches studies and returns the ID of the first match."""
        base_url = "https://www.cbioportal.org/api/studies"
        params = {'keyword': keyword, 'projection': 'SUMMARY', 'pageSize': 10, 'pageNumber': 0, 'direction': 'ASC'}
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            studies_list = response.json()
            if studies_list:
                return studies_list[0]['studyId']
        except requests.exceptions.RequestException:
            return None
        return None

    def _get_ids(self, study_id: str) -> Tuple[Optional[str], Optional[str]]:
        """Gets the mutation profile ID and sample list ID."""
        try:
            # Get molecular profiles
            profile_url = f"https://www.cbioportal.org/api/studies/{study_id}/molecular-profiles"
            profile_res = requests.get(profile_url)
            profile_res.raise_for_status()
            profiles = profile_res.json()
            mutation_profile_id = next((p['molecularProfileId'] for p in profiles if p['molecularAlterationType'] == 'MUTATION_EXTENDED'), None)

            # Get sample lists
            sample_list_url = f"https://www.cbioportal.org/api/studies/{study_id}/sample-lists"
            sample_res = requests.get(sample_list_url)
            sample_res.raise_for_status()
            sample_lists = sample_res.json()
            sample_list_id = next((s['sampleListId'] for s in sample_lists if '_all' in s['sampleListId']), None)
            if not sample_list_id and sample_lists:
                sample_list_id = sample_lists[0]['sampleListId']

            return mutation_profile_id, sample_list_id
        except requests.exceptions.RequestException:
            return None, None

    def _fetch_mutations(self, mutation_profile_id: str, sample_list_id: str, genes: List[str]) -> Optional[List[Dict[str, Any]]]:
        """Fetches mutation data."""
        url = f"https://www.cbioportal.org/api/molecular-profiles/{mutation_profile_id}/mutations"
        params = {'sampleListId': sample_list_id, 'geneList': ",".join(genes), 'projection': 'DETAILED'}
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException:
            return None

    def _get_total_sample_count(self, sample_list_id: str) -> int:
        """Gets the total number of samples."""
        url = f"https://www.cbioportal.org/api/sample-lists/{sample_list_id}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            return len(response.json().get('sampleIds', []))
        except requests.exceptions.RequestException:
            return 0

    def _calculate_gene_features(self, mutation_data: List[Dict[str, Any]], total_samples: int, gene_list: List[str]) -> Dict[str, Dict[str, Any]]:
        """Calculates summary features for each gene."""
        if not mutation_data or total_samples == 0:
            return {}
        mutations_by_gene = {gene: [] for gene in gene_list}
        for mut in mutation_data:
            gene_symbol = mut['gene']['hugoGeneSymbol']
            if gene_symbol in mutations_by_gene:
                mutations_by_gene[gene_symbol].append(mut)

        features = {}
        for gene, gene_mutations in mutations_by_gene.items():
            if not gene_mutations:
                features[gene] = {"mutation_frequency": 0.0, "compact_profile": "freq:0.00"}
                continue

            unique_patients = {mut['patientId'] for mut in gene_mutations}
            frequency = len(unique_patients) / total_samples

            vaf_sum, vaf_count = 0, 0
            for mut in gene_mutations:
                alt, ref = mut.get('tumorAltCount', 0), mut.get('tumorRefCount', 0)
                if (alt + ref) > 0:
                    vaf_sum += alt / (alt + ref)
                    vaf_count += 1
            avg_vaf = (vaf_sum / vaf_count) if vaf_count > 0 else 0

            top_type = Counter(mut['mutationType'] for mut in gene_mutations).most_common(1)[0][0]

            features[gene] = {
                "mutation_frequency": round(frequency, 4),
                "compact_profile": f"freq:{frequency:.2f}|avg_vaf:{avg_vaf:.2f}|top_type:{top_type}"
            }
        return features

    def forward(self, genes: str, cancer_name: str) -> str:
        """
        Executes the full workflow: parse query, fetch data, and calculate features.
        """
        gene_list = [gene.strip().upper() for gene in genes.split(',')]

        study_id = self._get_first_study_id(cancer_name)
        if not study_id:
            return f"Error: Could not find a study matching '{cancer_name}'."

        mutation_profile_id, sample_list_id = self._get_ids(study_id)
        if not mutation_profile_id or not sample_list_id:
            return f"Error: Could not find necessary mutation data or sample lists for study '{study_id}'."

        mutation_data = self._fetch_mutations(mutation_profile_id, sample_list_id, gene_list)
        if mutation_data is None:
            return "Error: Failed to fetch mutation data."

        total_samples = self._get_total_sample_count(sample_list_id)
        if total_samples == 0:
            return "Error: Could not determine the total number of samples for the study."

        gene_features = self._calculate_gene_features(mutation_data, total_samples, gene_list)

        if not gene_features:
            return "Could not calculate features. The genes might not have mutations in this study or an error occurred."

        return json.dumps(gene_features, indent=2)



# --- Helper function to build the compound map ---
def _build_compound_map_from_url(file_url: str) -> dict:
    """
    Fetches the DSigDB_All.txt file from the given URL and parses it into a dictionary (map)
    where keys are normalized compound names and values are sorted lists of gene names.
    """
    compound_to_genes_map = defaultdict(set)
    current_compound = None
    line_bytes = None # Initialize for potential error reporting

    def _normalize_compound_name_for_map(name: str) -> str:
        """
        Normalizes compound names by converting to lowercase and removing parenthetical
        identifiers or common database prefixes.
        """
        if not name:
            return ""
        name = name.lower()
        # Remove anything in parentheses (e.g., (CHEBI:XXXX), (CAS:XXXX))
        name = re.sub(r'\s*\(.*\)\s*$', '', name).strip()
        # Remove specific prefixes if they appear at the beginning of the name
        name = re.sub(r'^(chebi|cas|drugbank|pubchem|mesh|unii|hmdb):?\s*', '', name).strip()
        return name

    try:
        response = requests.get(file_url, stream=True) # Use stream=True for large files
        response.raise_for_status()

        # Process the file line by line
        for line_bytes_iter in response.iter_lines(decode_unicode=True):
            line_bytes = line_bytes_iter # Update for potential error reporting
            line = line_bytes.strip()
            if not line:
                continue

            # Check for compound header line
            compound_match = re.match(r'^Compound\(D\d+\)\s*:\s*(.*)$', line)
            if compound_match:
                raw_compound_name = compound_match.group(1).strip()
                current_compound = _normalize_compound_name_for_map(raw_compound_name)
                continue

            # If we have a current compound, and the line looks like a gene entry
            if current_compound:
                # Gene lines appear to be tab-separated: GeneSymbol \t InteractionType
                parts = line.split('\t')
                if len(parts) >= 1: # At least a gene symbol
                    gene_symbol = parts[0].strip()
                    if gene_symbol: # Ensure it's not an empty string
                        compound_to_genes_map[current_compound].add(gene_symbol)
        
        # Convert sets to sorted lists for consistent output and cache
        return {k: sorted(list(v)) for k, v in compound_to_genes_map.items()}

    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Could not fetch data from DSigDB URL: {e}")
    except Exception as e:
        raise ValueError(f"Error parsing DSigDB data: {e}. Last processed line: {line_bytes if line_bytes is not None else 'N/A'}")