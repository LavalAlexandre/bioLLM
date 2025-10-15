"""
cBioPortal data analysis package.

Organized modular structure for fetching and analyzing cancer genomics data.
"""

from .client import get_client, get_multiple_study_ids, get_genes, fetch_study_data
from .mutations import calculate_mutation_features
from .expression import calculate_expression_features, calculate_fold_changes
from .copy_number import calculate_cna_features
from .clinical import calculate_clinical_features
from .utils import safe_getattr

__all__ = [
    "get_client",
    "get_multiple_study_ids",
    "get_genes",
    "fetch_study_data",
    "calculate_mutation_features",
    "calculate_expression_features",
    "calculate_fold_changes",
    "calculate_cna_features",
    "calculate_clinical_features",
    "safe_getattr",
]
