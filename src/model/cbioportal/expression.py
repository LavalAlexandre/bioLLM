"""mRNA and protein expression analysis functions."""

from typing import List, Dict, Any
import numpy as np
from .utils import safe_getattr


def calculate_expression_features(
    expression_data: List[Any],
    gene_list: List[str],
    gene_id_to_symbol: Dict[int, str],
    is_normal: bool = False,
    is_protein: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Calculate expression statistics for each gene.

    Works for both mRNA and protein (RPPA) data.
    Values are typically z-scores or log2-normalized expression.

    Args:
        expression_data: List of expression objects from API
        gene_list: List of gene symbols
        gene_id_to_symbol: Mapping of Entrez IDs to symbols
        is_normal: Whether this is normal tissue data
        is_protein: Whether this is protein data (vs mRNA)

    Returns:
        Dictionary of gene -> features
    """
    if not expression_data:
        return {}

    # Group expression values by gene
    expression_by_gene = {}
    for item in expression_data:
        entrez_id = safe_getattr(item, "entrezGeneId")
        value = safe_getattr(item, "value", 0)

        if entrez_id:
            gene_symbol = gene_id_to_symbol.get(entrez_id, "").upper()
            if gene_symbol in [g.upper() for g in gene_list]:
                if gene_symbol not in expression_by_gene:
                    expression_by_gene[gene_symbol] = []
                try:
                    expression_by_gene[gene_symbol].append(float(value))
                except (ValueError, TypeError):
                    continue

    # Calculate features for each gene
    features = {}
    for gene in gene_list:
        gene_upper = gene.upper()
        if gene_upper not in expression_by_gene or not expression_by_gene[gene_upper]:
            continue

        values = np.array(expression_by_gene[gene_upper])

        # Basic statistics
        mean_expr = np.mean(values)
        median_expr = np.median(values)
        std_expr = np.std(values, ddof=1) if len(values) > 1 else 0

        if is_normal:
            # For normal samples, just return basic stats
            features[gene] = {
                "expression_profile": f"mean:{mean_expr:.2f}|median:{median_expr:.2f}|std:{std_expr:.2f}",
                "sample_count": len(values),
            }
        else:
            # For tumor samples, treat values as z-scores
            # Count altered samples (|z-score| > 2)
            altered_count = np.sum(np.abs(values) > 2)
            altered_pct = (altered_count / len(values)) * 100

            # Separate high and low
            high_count = np.sum(values > 2)
            low_count = np.sum(values < -2)
            high_pct = (high_count / len(values)) * 100
            low_pct = (low_count / len(values)) * 100

            # Quartiles for distribution shape
            q25 = np.percentile(values, 25)
            q75 = np.percentile(values, 75)

            features[gene] = {
                "expression_profile": f"mean:{mean_expr:.2f}|median:{median_expr:.2f}|std:{std_expr:.2f}|altered_pct:{altered_pct:.2f}",
                "z_score_profile": f"mean_z:{mean_expr:.2f}|high_pct:{high_pct:.2f}|low_pct:{low_pct:.2f}|q25:{q25:.2f}|q75:{q75:.2f}",
                "sample_count": len(values),
            }

    return features


def calculate_fold_changes(
    tumor_data: List[Any],
    normal_data: List[Any],
    gene_list: List[str],
    gene_id_to_symbol: Dict[int, str],
) -> Dict[str, Dict[str, Any]]:
    """
    Calculate fold changes between tumor and normal expression.

    Args:
        tumor_data: Tumor expression data
        normal_data: Normal tissue expression data
        gene_list: List of gene symbols
        gene_id_to_symbol: Mapping of Entrez IDs to symbols

    Returns:
        Dictionary of gene -> fold change features
    """

    def extract_values(data: List[Any]) -> Dict[str, np.ndarray]:
        values_by_gene = {}
        for item in data:
            entrez_id = safe_getattr(item, "entrezGeneId")
            value = safe_getattr(item, "value", 0)

            if entrez_id:
                gene_symbol = gene_id_to_symbol.get(entrez_id, "").upper()
                if gene_symbol in [g.upper() for g in gene_list]:
                    if gene_symbol not in values_by_gene:
                        values_by_gene[gene_symbol] = []
                    try:
                        values_by_gene[gene_symbol].append(float(value))
                    except (ValueError, TypeError):
                        continue
        return {k: np.array(v) for k, v in values_by_gene.items()}

    tumor_values = extract_values(tumor_data)
    normal_values = extract_values(normal_data)

    features = {}
    for gene in gene_list:
        gene_upper = gene.upper()
        if gene_upper not in tumor_values or gene_upper not in normal_values:
            continue

        if len(tumor_values[gene_upper]) == 0 or len(normal_values[gene_upper]) == 0:
            continue

        tumor_mean = np.mean(tumor_values[gene_upper])
        normal_mean = np.mean(normal_values[gene_upper])

        # Calculate log2 fold change (values are usually already log2 transformed)
        log2fc = tumor_mean - normal_mean

        # Count samples above/below normal mean
        up_count = np.sum(tumor_values[gene_upper] > normal_mean)
        down_count = np.sum(tumor_values[gene_upper] < normal_mean)
        up_pct = (up_count / len(tumor_values[gene_upper])) * 100
        down_pct = (down_count / len(tumor_values[gene_upper])) * 100

        # Effect size classification
        if abs(log2fc) < 0.5:
            effect = "minimal"
        elif abs(log2fc) < 1.0:
            effect = "moderate"
        else:
            effect = "strong"

        direction = "up" if log2fc > 0 else "down"

        features[gene] = {
            "fold_change_profile": f"log2fc:{log2fc:.2f}|{direction}|{effect}|up_pct:{up_pct:.2f}|down_pct:{down_pct:.2f}"
        }

    return features
