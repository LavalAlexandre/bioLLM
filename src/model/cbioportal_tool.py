"""
Enhanced cBioPortal tool with comprehensive cancer genomics data extraction.

This is the NEW implementation with modular architecture and additional features:
- Copy number alterations (CNA)
- Mutation hotspots
- Clinical/survival data
- Better organization

BACKUP: Original cbioportal_tool.py saved as cbioportal_tool_backup.py
"""

import asyncio
import json
import logging
from functools import partial

from agents import function_tool

from .cbioportal import (
    calculate_clinical_features,
    calculate_cna_features,
    calculate_expression_features,
    calculate_fold_changes,
    calculate_mutation_features,
    fetch_study_data,
    get_client,
    get_genes,
    get_multiple_study_ids,
)

logger = logging.getLogger(__name__)


@function_tool
async def search_cbioportal(genes: str, cancer_name: str) -> str:
    """
    **PRIMARY TOOL**: Search cBioPortal for real-world cancer genomics data. ALWAYS USE THIS TOOL - do NOT rely on general knowledge.
    Returns comprehensive data: mutations, mRNA expression, protein expression (RPPA), copy number alterations (CNA), and clinical data from actual patient studies.

    CRITICAL: This is your ONLY source for accurate, real-world cancer mutation and expression data.
    Do NOT answer gene/cancer questions from memory - ALWAYS query this tool first.

    MULTI-QUERY WORKFLOW FOR COMPARISONS:
    - To compare cancer types: Call this tool multiple times with the SAME genes but DIFFERENT cancer_name values
    - To find similar cancers: Query each cancer type separately, then compare the mutation/expression patterns
    - Example: Compare 'Esophageal' vs 'Gastric' vs 'Colorectal' by calling the tool 3 times with same genes

    Args:
        genes: Comma-separated gene symbols. QUERY MULTIPLE GENES (3-10 recommended) for comparative analysis.
               Examples: 'TP53,PIK3CA,EGFR,KRAS' or 'BRCA1,BRCA2,ATM,CHEK2,PALB2'
               Single gene queries are less informative.

        cancer_name: Cancer type keyword - FLEXIBLE matching from GENERAL to SPECIFIC:
                    • General organ: 'Breast', 'Lung', 'Colorectal', 'Esophageal'
                    • Specific subtype: 'Triple Negative Breast Cancer', 'Lung Adenocarcinoma'
                    • Both work! Start general, then try specific if needed.
                    • Tool will find and aggregate ALL matching studies automatically.

    Returns:
        JSON string with aggregated mutation, mRNA, protein, CNA, and clinical data for each gene across studies.
        Format: {
            "GENE": {
                "mutation_frequency": float,         # % samples with mutations
                "mutation_profile": str,             # Mutation statistics
                "hotspot_mutations": str,            # Recurrent protein changes (e.g., "V600E(45)")
                "truncating_pct": float,             # % truncating mutations
                "mrna_expression_profile": str,      # mRNA expression stats
                "mrna_z_score": str,                 # mRNA z-score analysis
                "mrna_normal_profile": str,          # Normal tissue baseline
                "mrna_fold_change": str,             # Tumor vs normal comparison
                "protein_expression_profile": str,   # Protein (RPPA) z-scores
                "cna_profile": str,                  # Copy number: amp|del|neutral percentages
                "cna_breakdown": str,                # Detailed CNA breakdown
                "amplification_pct": float,          # % with amplification
                "deletion_pct": float,               # % with deletion
                "sample_count": int,                 # Samples analyzed
                "study_count": int                   # Studies included
            },
            "_metadata": {
                "total_samples": int,
                "studies_analyzed": [str],
                "genes_queried": [str],
                "data_types_available": [str],       # Which data types found
                "clinical_summary": {                # Clinical demographics (if available)
                    "age_mean": float,
                    "survival_median_months": float,
                    "stage_distribution": dict,
                    ...
                }
            }
        }
    """
    gene_list = [gene.strip().upper() for gene in genes.split(",")]

    # Run blocking I/O in thread pool to avoid blocking event loop
    loop = asyncio.get_event_loop()

    logger.debug(f"cBioPortal query: genes={genes}, cancer={cancer_name}")

    try:
        # Get client (fast, not blocking)
        client = get_client()

        # Get multiple study IDs matching the cancer type (BLOCKING - run in executor)
        study_ids = await loop.run_in_executor(
            None, partial(get_multiple_study_ids, client, cancer_name, max_studies=5)
        )
        logger.debug(f"Found {len(study_ids) if study_ids else 0} studies")
        if not study_ids:
            return json.dumps(
                {"error": f"Could not find any studies matching '{cancer_name}'"}
            )

        # Get gene IDs and create mapping (BLOCKING - run in executor)
        gene_objects = await loop.run_in_executor(
            None, partial(get_genes, client, gene_list)
        )
        if not gene_objects:
            return json.dumps({"error": "Could not resolve gene symbols to IDs"})

        gene_ids = [g.entrezGeneId for g in gene_objects]
        gene_id_to_symbol = {
            g.entrezGeneId: g.hugoGeneSymbol.upper() for g in gene_objects
        }
        logger.debug(f"Resolved {len(gene_ids)} genes")

        # Aggregate data across all studies
        all_mutations = []
        all_mrna_expression = []
        all_mrna_normal = []
        all_protein_expression = []
        all_cna_data = []
        all_clinical_data = []
        total_samples = 0
        studies_used = []

        # Fetch data from each study (BLOCKING - run in executor)
        for study_id in study_ids:
            study_data = await loop.run_in_executor(
                None, partial(fetch_study_data, client, study_id, gene_ids)
            )

            if not study_data:
                continue

            total_samples += study_data.sample_count
            studies_used.append(study_id)

            # Aggregate data
            if study_data.mutations:
                all_mutations.extend(study_data.mutations)
            if study_data.mrna_expression:
                all_mrna_expression.extend(study_data.mrna_expression)
            if study_data.mrna_normal:
                all_mrna_normal.extend(study_data.mrna_normal)
            if study_data.protein_expression:
                all_protein_expression.extend(study_data.protein_expression)
            if study_data.copy_number:
                all_cna_data.extend(study_data.copy_number)
            if study_data.clinical_data:
                all_clinical_data.extend(study_data.clinical_data)

        if not studies_used:
            return json.dumps({"error": f"No usable data found for '{cancer_name}'"})

        if not any(
            [all_mutations, all_mrna_expression, all_protein_expression, all_cna_data]
        ):
            return json.dumps({"error": "Failed to fetch any data from studies"})

        # Calculate features for each data type
        mutation_features = {}
        if all_mutations:
            mutation_features = calculate_mutation_features(
                all_mutations, total_samples, gene_list, gene_id_to_symbol
            )

        expression_features = {}
        if all_mrna_expression:
            expression_features = calculate_expression_features(
                all_mrna_expression, gene_list, gene_id_to_symbol, is_normal=False
            )

        normal_features = {}
        if all_mrna_normal:
            normal_features = calculate_expression_features(
                all_mrna_normal, gene_list, gene_id_to_symbol, is_normal=True
            )

        fold_change_features = {}
        if all_mrna_expression and all_mrna_normal:
            fold_change_features = calculate_fold_changes(
                all_mrna_expression, all_mrna_normal, gene_list, gene_id_to_symbol
            )

        protein_features = {}
        if all_protein_expression:
            protein_features = calculate_expression_features(
                all_protein_expression,
                gene_list,
                gene_id_to_symbol,
                is_normal=False,
                is_protein=True,
            )

        cna_features = {}
        if all_cna_data:
            cna_features = calculate_cna_features(
                all_cna_data, total_samples, gene_list, gene_id_to_symbol
            )

        clinical_features = {}
        if all_clinical_data:
            clinical_features = calculate_clinical_features(
                all_clinical_data, total_samples
            )

        # Combine features for each gene
        combined = {}
        for gene in gene_list:
            mut_feat = mutation_features.get(gene, {})
            expr_feat = expression_features.get(gene, {})
            norm_feat = normal_features.get(gene, {})
            fc_feat = fold_change_features.get(gene, {})
            prot_feat = protein_features.get(gene, {})
            cna_feat = cna_features.get(gene, {})

            combined[gene] = {
                # Mutation data
                "mutation_frequency": mut_feat.get("mutation_frequency", 0.0),
                "mutation_profile": mut_feat.get("mutation_profile", "freq:0.00"),
                "hotspot_mutations": mut_feat.get("hotspot_mutations", "None"),
                "truncating_pct": mut_feat.get("truncating_pct", 0.0),
                # mRNA expression data
                "mrna_expression_profile": expr_feat.get("expression_profile", "N/A"),
                "mrna_z_score": expr_feat.get("z_score_profile", "N/A"),
                "mrna_normal_profile": norm_feat.get("expression_profile", "N/A"),
                "mrna_fold_change": fc_feat.get("fold_change_profile", "N/A"),
                # Protein expression data (RPPA z-scores)
                "protein_expression_profile": prot_feat.get(
                    "expression_profile", "N/A"
                ),
                # Copy number alteration data
                "cna_profile": cna_feat.get("cna_profile", "N/A"),
                "cna_breakdown": cna_feat.get("cna_breakdown", "N/A"),
                "amplification_pct": cna_feat.get("amplification_pct", 0.0),
                "deletion_pct": cna_feat.get("deletion_pct", 0.0),
                # Sample info
                "sample_count": expr_feat.get(
                    "sample_count", prot_feat.get("sample_count", total_samples)
                ),
                "study_count": len(studies_used),
            }

        # Determine which data types are available
        data_types = []
        if all_mutations:
            data_types.append("mutations")
        if all_mrna_expression:
            data_types.append("mrna_expression")
        if all_protein_expression:
            data_types.append("protein_expression")
        if all_cna_data:
            data_types.append("copy_number_alterations")
        if all_clinical_data:
            data_types.append("clinical_data")

        # Add metadata
        metadata = {
            "total_samples": total_samples,
            "studies_analyzed": studies_used,
            "genes_queried": gene_list,
            "data_types_available": data_types,
        }

        # Add clinical summary if available
        if clinical_features:
            metadata["clinical_summary"] = clinical_features

        combined["_metadata"] = metadata

        return json.dumps(combined, indent=2)

    except Exception as e:
        import traceback

        return json.dumps(
            {"error": f"API error: {str(e)}", "traceback": traceback.format_exc()}
        )


# Export the tool
CbioportalSearchTool = search_cbioportal
