from typing import List, Dict, Optional, Any
import requests
from collections import Counter
import json
from agents import function_tool


@function_tool
def search_cbioportal(genes: str, cancer_name: str) -> str:
    """
    Search cBioPortal to get gene mutation statistics for a specific cancer type.
    
    Args:
        genes: Comma-separated list of gene symbols (e.g., 'TP53,PIK3CA')
        cancer_name: Name of the cancer study (e.g., 'Breast Invasive Carcinoma')
        
    Returns:
        JSON string with mutation frequency and profile for each gene.
        Format: {"GENE": {"mutation_frequency": float, "compact_profile": str}}
    """
    gene_list = [gene.strip().upper() for gene in genes.split(',')]
    
    # Get study ID
    study_id = _get_study_id(cancer_name)
    if not study_id:
        return json.dumps({"error": f"Could not find study matching '{cancer_name}'"})
    
    # Get mutation profile and sample list IDs
    mutation_profile_id, sample_list_id = _get_profile_ids(study_id)
    if not mutation_profile_id or not sample_list_id:
        return json.dumps({"error": f"Missing mutation data for study '{study_id}'"})
    
    # Fetch mutation data
    mutation_data = _fetch_mutations(mutation_profile_id, sample_list_id, gene_list)
    if mutation_data is None:
        return json.dumps({"error": "Failed to fetch mutation data"})
    
    # Get total sample count
    total_samples = _get_sample_count(sample_list_id)
    if total_samples == 0:
        return json.dumps({"error": "Could not determine total sample count"})
    
    # Calculate gene features
    gene_features = _calculate_gene_features(mutation_data, total_samples, gene_list)
    
    if not gene_features:
        return json.dumps({"error": "No mutation data found for specified genes"})
    
    return json.dumps(gene_features, indent=2)


def _get_study_id(keyword: str) -> Optional[str]:
    """Fetch the first matching study ID from cBioPortal."""
    url = "https://www.cbioportal.org/api/studies"
    params = {
        'keyword': keyword,
        'projection': 'SUMMARY',
        'pageSize': 10,
        'pageNumber': 0,
        'direction': 'ASC'
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        studies = response.json()
        return studies[0]['studyId'] if studies else None
    except requests.exceptions.RequestException:
        return None


def _get_profile_ids(study_id: str) -> tuple[Optional[str], Optional[str]]:
    """Get mutation profile ID and sample list ID for a study."""
    try:
        # Get molecular profiles
        profile_url = f"https://www.cbioportal.org/api/studies/{study_id}/molecular-profiles"
        profile_res = requests.get(profile_url, timeout=10)
        profile_res.raise_for_status()
        profiles = profile_res.json()
        
        mutation_profile_id = next(
            (p['molecularProfileId'] for p in profiles 
             if p['molecularAlterationType'] == 'MUTATION_EXTENDED'),
            None
        )
        
        # Get sample lists
        sample_url = f"https://www.cbioportal.org/api/studies/{study_id}/sample-lists"
        sample_res = requests.get(sample_url, timeout=10)
        sample_res.raise_for_status()
        sample_lists = sample_res.json()
        
        # Prefer '_all' sample list, fallback to first available
        sample_list_id = next(
            (s['sampleListId'] for s in sample_lists if '_all' in s['sampleListId']),
            sample_lists[0]['sampleListId'] if sample_lists else None
        )
        
        return mutation_profile_id, sample_list_id
        
    except requests.exceptions.RequestException:
        return None, None


def _fetch_mutations(
    mutation_profile_id: str,
    sample_list_id: str,
    genes: List[str]
) -> Optional[List[Dict[str, Any]]]:
    """Fetch mutation data for specified genes."""
    url = f"https://www.cbioportal.org/api/molecular-profiles/{mutation_profile_id}/mutations"
    params = {
        'sampleListId': sample_list_id,
        'geneList': ",".join(genes),
        'projection': 'DETAILED'
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return None


def _get_sample_count(sample_list_id: str) -> int:
    """Get total number of samples in the sample list."""
    url = f"https://www.cbioportal.org/api/sample-lists/{sample_list_id}"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return len(response.json().get('sampleIds', []))
    except requests.exceptions.RequestException:
        return 0


def _calculate_gene_features(
    mutation_data: List[Dict[str, Any]],
    total_samples: int,
    gene_list: List[str]
) -> Dict[str, Dict[str, Any]]:
    """Calculate mutation frequency and profile for each gene."""
    if not mutation_data or total_samples == 0:
        return {}
    
    # Group mutations by gene
    mutations_by_gene = {gene: [] for gene in gene_list}
    for mutation in mutation_data:
        gene_symbol = mutation['gene']['hugoGeneSymbol']
        if gene_symbol in mutations_by_gene:
            mutations_by_gene[gene_symbol].append(mutation)
    
    # Calculate features for each gene
    features = {}
    for gene, gene_mutations in mutations_by_gene.items():
        if not gene_mutations:
            features[gene] = {
                "mutation_frequency": 0.0,
                "compact_profile": "freq:0.00"
            }
            continue
        
        # Calculate mutation frequency
        unique_patients = {mut['patientId'] for mut in gene_mutations}
        frequency = len(unique_patients) / total_samples
        
        # Calculate average VAF (Variant Allele Frequency)
        vaf_values = []
        for mut in gene_mutations:
            alt = mut.get('tumorAltCount', 0)
            ref = mut.get('tumorRefCount', 0)
            total = alt + ref
            if total > 0:
                vaf_values.append(alt / total)
        
        avg_vaf = sum(vaf_values) / len(vaf_values) if vaf_values else 0
        
        # Get most common mutation type
        mutation_types = [mut['mutationType'] for mut in gene_mutations]
        top_type = Counter(mutation_types).most_common(1)[0][0]
        
        features[gene] = {
            "mutation_frequency": round(frequency, 4),
            "compact_profile": f"freq:{frequency:.2f}|avg_vaf:{avg_vaf:.2f}|top_type:{top_type}"
        }
    
    return features


# Export the tool
CbioportalSearchTool = search_cbioportal