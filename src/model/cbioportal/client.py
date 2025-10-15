"""cBioPortal API client and data fetching functions."""

from typing import List, Optional, Any, Tuple
from bravado.client import SwaggerClient
from bravado.requests_client import RequestsClient
import requests.adapters


# Singleton client instance
_cbioportal_client = None


def get_client() -> SwaggerClient:
    """Get or create the cBioPortal swagger client with optimized connection pool."""
    global _cbioportal_client
    if _cbioportal_client is None:
        # Configure HTTP client with larger connection pool
        # Default pool size is 10, but we have 80 concurrent requests
        # Set to 150 to handle burst traffic from concurrent tool calls
        http_client = RequestsClient()
        http_client.session.mount(
            "https://",
            requests.adapters.HTTPAdapter(
                pool_connections=150,  # Number of connection pools
                pool_maxsize=150,  # Max connections per pool
                max_retries=3,
                pool_block=False,
            ),
        )
        http_client.session.mount(
            "http://",
            requests.adapters.HTTPAdapter(
                pool_connections=150, pool_maxsize=150, max_retries=3, pool_block=False
            ),
        )

        _cbioportal_client = SwaggerClient.from_url(
            "https://www.cbioportal.org/api/v2/api-docs",
            http_client=http_client,
            config={
                "validate_requests": False,
                "validate_responses": False,
                "validate_swagger_spec": False,
            },
        )
    return _cbioportal_client


def get_multiple_study_ids(
    client: SwaggerClient, keyword: str, max_studies: int = 5
) -> List[str]:
    """
    Fetch multiple matching study IDs from cBioPortal.

    Args:
        client: Bravado client instance
        keyword: Cancer type keyword to search for
        max_studies: Maximum number of studies to return

    Returns:
        List of study IDs, prioritizing larger studies
    """
    try:
        studies = client.Studies.getAllStudiesUsingGET(
            keyword=keyword, projection="SUMMARY", pageSize=20, pageNumber=0
        ).result()

        if not studies:
            return []

        # Sort by sample count (larger studies first) and limit
        sorted_studies = sorted(
            studies,
            key=lambda s: s.allSampleCount if hasattr(s, "allSampleCount") else 0,
            reverse=True,
        )

        return [s.studyId for s in sorted_studies[:max_studies]]
    except Exception:
        return []


def get_genes(client: SwaggerClient, gene_symbols: List[str]) -> List[Any]:
    """Get gene objects for gene symbols."""
    try:
        genes = client.Genes.fetchGenesUsingPOST(
            geneIdType="HUGO_GENE_SYMBOL", geneIds=gene_symbols
        ).result()

        return list(genes) if genes else []
    except Exception:
        return []


def get_profile_ids(client: SwaggerClient, study_id: str) -> Tuple[Optional[str], ...]:
    """
    Get molecular profile IDs and sample list ID for a study.

    Returns:
        Tuple of (mutation_profile, mrna_profile, protein_profile, cna_profile, sample_list_id)
    """
    try:
        # Get molecular profiles
        profiles = client.Molecular_Profiles.getAllMolecularProfilesInStudyUsingGET(
            studyId=study_id, projection="SUMMARY"
        ).result()

        mutation_profile_id = next(
            (
                p.molecularProfileId
                for p in profiles
                if p.molecularAlterationType == "MUTATION_EXTENDED"
            ),
            None,
        )

        # mRNA expression profiles
        expression_profile_id = next(
            (
                p.molecularProfileId
                for p in profiles
                if p.molecularAlterationType == "MRNA_EXPRESSION"
                and (
                    "rna_seq" in p.molecularProfileId.lower()
                    or "mrna" in p.molecularProfileId.lower()
                )
            ),
            next(
                (
                    p.molecularProfileId
                    for p in profiles
                    if p.molecularAlterationType == "MRNA_EXPRESSION"
                ),
                None,
            ),
        )

        # Protein expression profiles (RPPA)
        protein_profile_id = next(
            (
                p.molecularProfileId
                for p in profiles
                if p.molecularAlterationType
                in ["PROTEIN_LEVEL", "PROTEIN_ARRAY_PROTEIN_LEVEL"]
            ),
            None,
        )

        # Copy number alteration profiles (CNA)
        cna_profile_id = next(
            (
                p.molecularProfileId
                for p in profiles
                if p.molecularAlterationType == "COPY_NUMBER_ALTERATION"
                and (
                    "gistic" in p.molecularProfileId.lower()
                    or "cna" in p.molecularProfileId.lower()
                )
            ),
            next(
                (
                    p.molecularProfileId
                    for p in profiles
                    if p.molecularAlterationType == "COPY_NUMBER_ALTERATION"
                ),
                None,
            ),
        )

        # Get sample lists
        sample_lists = client.Sample_Lists.getAllSampleListsInStudyUsingGET(
            studyId=study_id
        ).result()

        # Prefer RNA seq sample list, then '_all'
        sample_list_id = next(
            (
                s.sampleListId
                for s in sample_lists
                if "rna_seq" in s.sampleListId.lower()
            ),
            next(
                (s.sampleListId for s in sample_lists if "_all" in s.sampleListId),
                sample_lists[0].sampleListId if sample_lists else None,
            ),
        )

        return (
            mutation_profile_id,
            expression_profile_id,
            protein_profile_id,
            cna_profile_id,
            sample_list_id,
        )

    except Exception:
        return None, None, None, None, None


def get_normal_sample_list(client: SwaggerClient, study_id: str) -> Optional[str]:
    """Get normal/control sample list ID if available."""
    try:
        sample_lists = client.Sample_Lists.getAllSampleListsInStudyUsingGET(
            studyId=study_id
        ).result()

        # Look for normal tissue sample list
        normal_list_id = next(
            (
                s.sampleListId
                for s in sample_lists
                if any(
                    keyword in s.sampleListId.lower()
                    for keyword in ["normal", "control", "adj", "solid_tissue_normal"]
                )
            ),
            None,
        )

        return normal_list_id

    except Exception:
        return None


def get_sample_count(client: SwaggerClient, sample_list_id: str) -> int:
    """Get total number of samples in the sample list."""
    try:
        sample_list = client.Sample_Lists.getSampleListUsingGET(
            sampleListId=sample_list_id
        ).result()

        return len(sample_list.sampleIds) if sample_list.sampleIds else 0
    except Exception:
        return 0


def fetch_mutations(
    client: SwaggerClient,
    mutation_profile_id: str,
    sample_list_id: str,
    gene_ids: List[int],
) -> Optional[List[Any]]:
    """Fetch mutation data for specified genes."""
    try:
        result = client.Mutations.fetchMutationsInMolecularProfileUsingPOST(
            molecularProfileId=mutation_profile_id,
            mutationFilter={"sampleListId": sample_list_id, "entrezGeneIds": gene_ids},
            projection="DETAILED",  # Changed to DETAILED for protein changes
        ).result()

        return list(result) if result else None
    except Exception:
        return None


def fetch_molecular_data(
    client: SwaggerClient, profile_id: str, sample_list_id: str, gene_ids: List[int]
) -> Optional[List[Any]]:
    """
    Fetch molecular data (expression, CNA, etc.) for specified genes.
    Generic function for all molecular data types.
    """
    try:
        result = client.Molecular_Data.fetchAllMolecularDataInMolecularProfileUsingPOST(
            molecularProfileId=profile_id,
            molecularDataFilter={
                "sampleListId": sample_list_id,
                "entrezGeneIds": gene_ids,
            },
            projection="SUMMARY",
        ).result()

        return list(result) if result else None
    except Exception:
        return None


def fetch_clinical_data(
    client: SwaggerClient, study_id: str, sample_ids: Optional[List[str]] = None
) -> Optional[List[Any]]:
    """
    Fetch clinical data for patients in a study.

    Args:
        client: Bravado client
        study_id: Study ID
        sample_ids: Optional list of specific sample IDs

    Returns:
        List of clinical data objects or None
    """
    try:
        if sample_ids:
            # Fetch for specific samples
            result = client.Clinical_Data.fetchClinicalDataUsingPOST(
                clinicalDataType="PATIENT",
                clinicalDataMultiStudyFilter={
                    "identifiers": [
                        {"studyId": study_id, "entityId": sid} for sid in sample_ids
                    ]
                },
            ).result()
        else:
            # Fetch all clinical data for study
            result = client.Clinical_Data.getAllClinicalDataInStudyUsingGET(
                studyId=study_id, clinicalDataType="PATIENT", projection="SUMMARY"
            ).result()

        return list(result) if result else None
    except Exception:
        return None


class StudyData:
    """Container for all data fetched from a single study."""

    def __init__(self, study_id: str, sample_count: int):
        self.study_id = study_id
        self.sample_count = sample_count
        self.mutations = []
        self.mrna_expression = []
        self.mrna_normal = []
        self.protein_expression = []
        self.copy_number = []
        self.clinical_data = []

    def __repr__(self):
        return f"StudyData(study={self.study_id}, samples={self.sample_count})"


def fetch_study_data(
    client: SwaggerClient, study_id: str, gene_ids: List[int]
) -> Optional[StudyData]:
    """
    Fetch all available data types for a study.

    Args:
        client: Bravado client
        study_id: Study ID
        gene_ids: List of Entrez gene IDs

    Returns:
        StudyData object with all fetched data, or None if no data available
    """
    # Get profile IDs
    mutation_prof, mrna_prof, protein_prof, cna_prof, sample_list = get_profile_ids(
        client, study_id
    )

    if not sample_list:
        return None

    # Get sample count
    sample_count = get_sample_count(client, sample_list)
    if sample_count == 0:
        return None

    # Create container
    data = StudyData(study_id, sample_count)

    # Fetch mutations
    if mutation_prof:
        mutations = fetch_mutations(client, mutation_prof, sample_list, gene_ids)
        if mutations:
            data.mutations = mutations

    # Fetch mRNA expression
    if mrna_prof:
        mrna_expr = fetch_molecular_data(client, mrna_prof, sample_list, gene_ids)
        if mrna_expr:
            data.mrna_expression = mrna_expr

        # Try to fetch normal tissue mRNA
        normal_list = get_normal_sample_list(client, study_id)
        if normal_list:
            mrna_normal = fetch_molecular_data(client, mrna_prof, normal_list, gene_ids)
            if mrna_normal:
                data.mrna_normal = mrna_normal

    # Fetch protein expression (RPPA)
    if protein_prof:
        protein_expr = fetch_molecular_data(client, protein_prof, sample_list, gene_ids)
        if protein_expr:
            data.protein_expression = protein_expr

    # Fetch copy number alterations (CNA)
    if cna_prof:
        cna_data = fetch_molecular_data(client, cna_prof, sample_list, gene_ids)
        if cna_data:
            data.copy_number = cna_data

    # Fetch clinical data
    clinical = fetch_clinical_data(client, study_id)
    if clinical:
        data.clinical_data = clinical

    return data
