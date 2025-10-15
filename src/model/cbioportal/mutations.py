"""Mutation data analysis functions."""

from typing import List, Dict, Any
from collections import Counter
import numpy as np
from .utils import safe_getattr


def calculate_mutation_features(
    mutation_data: List[Any],
    total_samples: int,
    gene_list: List[str],
    gene_id_to_symbol: Dict[int, str],
) -> Dict[str, Dict[str, Any]]:
    """
    Calculate comprehensive mutation features for each gene.

    Features extracted:
    - Mutation frequency (% samples mutated)
    - Average VAF (variant allele frequency)
    - Mutation types distribution
    - Protein changes / hotspots
    - Truncating mutation percentage

    Args:
        mutation_data: List of mutation objects from API
        total_samples: Total number of samples
        gene_list: List of gene symbols
        gene_id_to_symbol: Mapping of Entrez IDs to symbols

    Returns:
        Dictionary of gene -> features
    """
    if not mutation_data:
        return {}

    # Group mutations by gene
    mutations_by_gene = {gene: [] for gene in gene_list}
    for mutation in mutation_data:
        entrez_id = safe_getattr(mutation, "entrezGeneId")

        if entrez_id:
            gene_symbol = gene_id_to_symbol.get(entrez_id, "").upper()
            if gene_symbol in mutations_by_gene:
                mutations_by_gene[gene_symbol].append(mutation)

    # Calculate features for each gene
    features = {}
    for gene, gene_mutations in mutations_by_gene.items():
        if not gene_mutations:
            features[gene] = {
                "mutation_frequency": 0.0,
                "mutation_profile": "freq:0.00",
                "hotspot_mutations": "None",
                "truncating_pct": 0.0,
            }
            continue

        # Calculate mutation frequency
        unique_patients = set()
        for mut in gene_mutations:
            patient_id = safe_getattr(mut, "patientId")
            if patient_id:
                unique_patients.add(patient_id)

        frequency = len(unique_patients) / total_samples if total_samples > 0 else 0

        # Calculate average VAF
        vaf_values = []
        for mut in gene_mutations:
            alt = safe_getattr(mut, "tumorAltCount", 0) or 0
            ref = safe_getattr(mut, "tumorRefCount", 0) or 0

            total = alt + ref
            if total > 0:
                vaf_values.append(alt / total)

        avg_vaf = float(np.mean(vaf_values)) if vaf_values else 0.0

        # Mutation types
        mutation_types = []
        for mut in gene_mutations:
            mut_type = safe_getattr(mut, "mutationType", "Unknown")
            mutation_types.append(mut_type)

        type_counter = Counter(mutation_types)
        # Note: top_type calculated but not currently used in output
        # top_type = type_counter.most_common(1)[0][0] if mutation_types else "Unknown"

        # Count truncating mutations (nonsense, frameshift, splice)
        truncating_types = {
            "Nonsense_Mutation",
            "Frame_Shift_Del",
            "Frame_Shift_Ins",
            "Splice_Site",
            "Translation_Start_Site",
        }
        truncating_count = sum(1 for t in mutation_types if t in truncating_types)
        truncating_pct = (
            (truncating_count / len(gene_mutations)) * 100 if gene_mutations else 0
        )

        # Extract protein changes (hotspots)
        protein_changes = []
        for mut in gene_mutations:
            protein_change = safe_getattr(mut, "proteinChange")
            if protein_change and protein_change != "NA":
                # Clean up protein change (e.g., "p.V600E" -> "V600E")
                cleaned = protein_change.replace("p.", "").strip()
                if cleaned:
                    protein_changes.append(cleaned)

        # Find hotspot mutations (appearing multiple times)
        hotspots = []
        if protein_changes:
            change_counter = Counter(protein_changes)
            # Consider mutations appearing in >2% of cases as hotspots
            hotspot_threshold = max(2, len(gene_mutations) * 0.02)
            hotspots = [
                f"{change}({count})"
                for change, count in change_counter.most_common(5)
                if count >= hotspot_threshold
            ]

        hotspot_str = "|".join(hotspots[:3]) if hotspots else "None"

        # Type distribution (top 3)
        type_dist = "|".join([f"{t}:{c}" for t, c in type_counter.most_common(3)])

        features[gene] = {
            "mutation_frequency": round(frequency, 4),
            "mutation_profile": f"freq:{frequency:.2f}|vaf:{avg_vaf:.2f}|types:{type_dist}",
            "hotspot_mutations": hotspot_str,
            "truncating_pct": round(truncating_pct, 2),
        }

    return features
