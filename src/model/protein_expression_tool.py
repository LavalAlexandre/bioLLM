import json
from agents import function_tool

# Import the comprehensive tool
from .cbioportal_tool import search_cbioportal as _search_cbioportal


@function_tool
def search_protein_expression(proteins: str, cancer_name: str) -> str:
    """
    **USE SEARCH_CBIOPORTAL INSTEAD** - This is a legacy wrapper for protein-only queries.

    For most use cases, call search_cbioportal() instead to get mutations + mRNA + protein in one query.
    Only use this if you SPECIFICALLY need protein-only output.

    CRITICAL: Do NOT answer protein expression questions from memory - ALWAYS query real data.

    MULTI-QUERY WORKFLOW FOR COMPARISONS:
    - To compare cancer types: Call this tool multiple times with SAME proteins but DIFFERENT cancer_name values
    - Example: Compare 'Breast' vs 'Ovarian' protein patterns by calling twice

    Args:
        proteins: Comma-separated protein/gene symbols. QUERY MULTIPLE (3-10 recommended).
                 Examples: 'AKT,EGFR,TP53,PTEN' or 'BRCA1,BRCA2,ATM,CHEK2'

        cancer_name: Cancer type keyword - FLEXIBLE matching from GENERAL to SPECIFIC:
                    • General: 'Breast', 'Lung', 'Colorectal'
                    • Specific: 'Triple Negative Breast Cancer', 'Lung Adenocarcinoma'
                    • Start general, refine if needed. Tool auto-aggregates all matching studies.

    Returns:
        JSON string with protein expression statistics extracted from the comprehensive tool.
        Format: {
            "PROTEIN": {
                "protein_expression_profile": str,    # z-score statistics (mean|median|altered_pct)
                "sample_count": int,                  # Total samples analyzed
                "study_count": int                    # Number of studies included
            },
            "_metadata": {
                "total_samples": int,
                "studies_analyzed": [str],
                "proteins_queried": [str],
                "data_types_available": [str]
            }
        }
    """
    # Call the comprehensive tool
    full_result = _search_cbioportal(genes=proteins, cancer_name=cancer_name)

    try:
        full_data = json.loads(full_result)

        # Check for error
        if "error" in full_data:
            return full_result

        # Extract only protein-related fields
        protein_only_data = {}
        metadata = full_data.get("_metadata", {})

        for protein in metadata.get("genes_queried", []):
            if protein in full_data:
                gene_data = full_data[protein]
                protein_only_data[protein] = {
                    "protein_expression_profile": gene_data.get(
                        "protein_expression_profile", "N/A"
                    ),
                    "sample_count": gene_data.get("sample_count", 0),
                    "study_count": gene_data.get("study_count", 0),
                }

        # Update metadata
        protein_only_data["_metadata"] = {
            "total_samples": metadata.get("total_samples", 0),
            "studies_analyzed": metadata.get("studies_analyzed", []),
            "proteins_queried": metadata.get("genes_queried", []),
            "data_types_available": metadata.get("data_types_available", []),
        }

        return json.dumps(protein_only_data, indent=2)

    except Exception as e:
        return json.dumps({"error": f"Error processing results: {str(e)}"})


# Export the tool
ProteinExpressionTool = search_protein_expression
