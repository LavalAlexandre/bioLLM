"""Clinical data analysis functions."""

from typing import List, Dict, Any
import numpy as np
from .utils import safe_getattr


def calculate_clinical_features(
    clinical_data: List[Any], sample_count: int
) -> Dict[str, Any]:
    """
    Calculate clinical/demographic features from patient data.

    Features extracted:
    - Age distribution (mean, median, range)
    - Overall survival statistics (if available)
    - Disease-free survival (if available)
    - Cancer stage distribution
    - Grade distribution
    - Sample demographics

    Args:
        clinical_data: List of clinical data objects from API
        sample_count: Total number of samples in the study

    Returns:
        Dictionary of clinical features
    """
    if not clinical_data:
        return {}

    features = {
        "total_patients": len(
            set(
                safe_getattr(d, "patientId")
                for d in clinical_data
                if safe_getattr(d, "patientId")
            )
        ),
        "clinical_attributes_available": [],
    }

    # Group by attribute type
    attributes = {}
    for item in clinical_data:
        attr_id = safe_getattr(item, "clinicalAttributeId")
        value = safe_getattr(item, "value")

        if attr_id and value:
            if attr_id not in attributes:
                attributes[attr_id] = []
            attributes[attr_id].append(value)

    # Extract age information
    age_attr = None
    for possible_age_attr in [
        "AGE",
        "AGE_AT_DIAGNOSIS",
        "AGE_AT_SEQUENCING",
        "PATIENT_AGE",
    ]:
        if possible_age_attr in attributes:
            age_attr = possible_age_attr
            break

    if age_attr:
        ages = []
        for val in attributes[age_attr]:
            try:
                age = float(val)
                if 0 < age < 120:  # Sanity check
                    ages.append(age)
            except (ValueError, TypeError):
                continue

        if ages:
            features["age_mean"] = round(np.mean(ages), 1)
            features["age_median"] = round(np.median(ages), 1)
            features["age_range"] = f"{int(min(ages))}-{int(max(ages))}"
            features["age_std"] = round(np.std(ages, ddof=1), 1) if len(ages) > 1 else 0
            features["clinical_attributes_available"].append("age")

    # Extract survival information
    os_months = None
    os_status = None

    for os_attr in ["OS_MONTHS", "OVERALL_SURVIVAL_MONTHS", "SURVIVAL_MONTHS"]:
        if os_attr in attributes:
            os_months = os_attr
            break

    for status_attr in ["OS_STATUS", "OVERALL_SURVIVAL_STATUS", "VITAL_STATUS"]:
        if status_attr in attributes:
            os_status = status_attr
            break

    if os_months:
        survival_times = []
        for val in attributes[os_months]:
            try:
                months = float(val)
                if months >= 0:  # Sanity check
                    survival_times.append(months)
            except (ValueError, TypeError):
                continue

        if survival_times:
            features["survival_median_months"] = round(np.median(survival_times), 1)
            features["survival_mean_months"] = round(np.mean(survival_times), 1)
            features["survival_range_months"] = (
                f"{min(survival_times):.1f}-{max(survival_times):.1f}"
            )
            features["patients_with_survival_data"] = len(survival_times)
            features["clinical_attributes_available"].append("survival")

    if os_status:
        statuses = attributes[os_status]
        deceased_keywords = ["DECEASED", "DEAD", "1:DECEASED", "Dead"]
        deceased_count = sum(
            1
            for s in statuses
            if any(keyword in str(s).upper() for keyword in deceased_keywords)
        )
        if len(statuses) > 0:
            features["mortality_rate"] = round(
                (deceased_count / len(statuses)) * 100, 1
            )

    # Extract stage information
    stage_attr = None
    for possible_stage in [
        "STAGE",
        "TUMOR_STAGE",
        "AJCC_STAGE",
        "PATHOLOGIC_STAGE",
        "CLINICAL_STAGE",
    ]:
        if possible_stage in attributes:
            stage_attr = possible_stage
            break

    if stage_attr:
        stages = [str(s).upper() for s in attributes[stage_attr]]
        # Count by broad categories
        stage_counts = {
            "I": sum(1 for s in stages if "STAGE I" in s or s in ["I", "IA", "IB"]),
            "II": sum(
                1 for s in stages if "STAGE II" in s or s in ["II", "IIA", "IIB", "IIC"]
            ),
            "III": sum(
                1
                for s in stages
                if "STAGE III" in s or s in ["III", "IIIA", "IIIB", "IIIC"]
            ),
            "IV": sum(
                1 for s in stages if "STAGE IV" in s or s in ["IV", "IVA", "IVB", "IVC"]
            ),
        }

        total_staged = sum(stage_counts.values())
        if total_staged > 0:
            stage_dist = {
                f"stage_{k}": f"{v} ({(v / total_staged) * 100:.1f}%)"
                for k, v in stage_counts.items()
                if v > 0
            }
            features["stage_distribution"] = stage_dist
            features["patients_with_stage_data"] = total_staged
            features["clinical_attributes_available"].append("stage")

    # Extract grade information
    grade_attr = None
    for possible_grade in ["GRADE", "TUMOR_GRADE", "HISTOLOGICAL_GRADE"]:
        if possible_grade in attributes:
            grade_attr = possible_grade
            break

    if grade_attr:
        grades = [str(g).upper() for g in attributes[grade_attr]]
        grade_counts = {}
        for grade in ["1", "2", "3", "4", "G1", "G2", "G3", "G4"]:
            count = sum(1 for g in grades if grade in g)
            if count > 0:
                grade_counts[grade] = count

        if grade_counts:
            total_graded = sum(grade_counts.values())
            grade_dist = {
                f"grade_{k}": f"{v} ({(v / total_graded) * 100:.1f}%)"
                for k, v in grade_counts.items()
            }
            features["grade_distribution"] = grade_dist
            features["patients_with_grade_data"] = total_graded
            features["clinical_attributes_available"].append("grade")

    # Add summary
    features["clinical_data_completeness"] = (
        f"{len(features['clinical_attributes_available'])} attributes available"
    )

    return features
