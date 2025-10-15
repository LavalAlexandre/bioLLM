"""Copy number alteration (CNA) analysis functions."""

from typing import List, Dict, Any
import numpy as np
from .utils import safe_getattr


def calculate_cna_features(
    cna_data: List[Any],
    total_samples: int,
    gene_list: List[str],
    gene_id_to_symbol: Dict[int, str],
) -> Dict[str, Dict[str, Any]]:
    """
    Calculate copy number alteration features for each gene.

    CNA values in cBioPortal (GISTIC):
    - -2: Deep deletion (homozygous deletion)
    - -1: Shallow deletion (heterozygous deletion)
    -  0: Diploid / neutral
    -  1: Low-level gain
    -  2: High-level amplification

    Features extracted:
    - Amplification frequency (value >= 2)
    - Gain frequency (value >= 1)
    - Deletion frequency (value <= -1)
    - Deep deletion frequency (value <= -2)
    - Mean copy number level

    Args:
        cna_data: List of CNA objects from API
        total_samples: Total number of samples
        gene_list: List of gene symbols
        gene_id_to_symbol: Mapping of Entrez IDs to symbols

    Returns:
        Dictionary of gene -> CNA features
    """
    if not cna_data:
        return {}

    # Group CNA values by gene
    cna_by_gene = {}
    for item in cna_data:
        entrez_id = safe_getattr(item, "entrezGeneId")
        value = safe_getattr(item, "value")

        if entrez_id and value is not None:
            gene_symbol = gene_id_to_symbol.get(entrez_id, "").upper()
            if gene_symbol in [g.upper() for g in gene_list]:
                if gene_symbol not in cna_by_gene:
                    cna_by_gene[gene_symbol] = []
                try:
                    cna_by_gene[gene_symbol].append(int(value))
                except (ValueError, TypeError):
                    continue

    # Calculate features for each gene
    features = {}
    for gene in gene_list:
        gene_upper = gene.upper()
        if gene_upper not in cna_by_gene or not cna_by_gene[gene_upper]:
            # No CNA data for this gene
            features[gene] = {
                "cna_profile": "N/A",
                "amplification_pct": 0.0,
                "deletion_pct": 0.0,
                "cna_alteration_pct": 0.0,
            }
            continue

        values = np.array(cna_by_gene[gene_upper])
        n_samples = len(values)

        # Count different alteration types
        deep_deletion = np.sum(values == -2)
        shallow_deletion = np.sum(values == -1)
        neutral = np.sum(values == 0)
        gain = np.sum(values == 1)
        amplification = np.sum(values == 2)

        # Calculate percentages
        deep_del_pct = (deep_deletion / n_samples) * 100
        shallow_del_pct = (shallow_deletion / n_samples) * 100
        any_deletion_pct = ((deep_deletion + shallow_deletion) / n_samples) * 100

        gain_pct = (gain / n_samples) * 100
        amp_pct = (amplification / n_samples) * 100
        any_amplification_pct = ((gain + amplification) / n_samples) * 100

        neutral_pct = (neutral / n_samples) * 100

        # Any alteration (non-neutral)
        altered = n_samples - neutral
        altered_pct = (altered / n_samples) * 100

        # Mean copy number level
        mean_cna = np.mean(values)

        # Determine dominant alteration
        if amplification > max(deep_deletion, shallow_deletion, gain, neutral):
            dominant = "high_amp"
        elif gain > max(deep_deletion, shallow_deletion, neutral):
            dominant = "gain"
        elif deep_deletion > max(shallow_deletion, neutral):
            dominant = "deep_del"
        elif shallow_deletion > neutral:
            dominant = "shallow_del"
        else:
            dominant = "neutral"

        # Create compact profile string
        cna_profile = f"amp:{amp_pct:.1f}%|del:{any_deletion_pct:.1f}%|neutral:{neutral_pct:.1f}%|dominant:{dominant}"

        # Detailed breakdown
        cna_breakdown = f"deep_del:{deep_del_pct:.1f}%|shallow_del:{shallow_del_pct:.1f}%|neutral:{neutral_pct:.1f}%|gain:{gain_pct:.1f}%|high_amp:{amp_pct:.1f}%"

        features[gene] = {
            "cna_profile": cna_profile,
            "cna_breakdown": cna_breakdown,
            "amplification_pct": round(any_amplification_pct, 2),
            "deletion_pct": round(any_deletion_pct, 2),
            "cna_alteration_pct": round(altered_pct, 2),
            "mean_cna_level": round(mean_cna, 2),
        }

    return features
