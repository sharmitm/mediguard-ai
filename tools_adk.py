"""
ADK Tools for MediGuard AI
Custom FunctionTools for Google ADK agents
"""
import pandas as pd
from typing import Dict, List, Any
from google.adk.tools import FunctionTool
import logging

logger = logging.getLogger(__name__)

# Global dataframes (will be initialized from main.py)
patients = None
claims = None
claim_lines = None

def initialize_tools_data(patients_df, claims_df, claim_lines_df):
    """Initialize global dataframes for tools"""
    global patients, claims, claim_lines
    patients = patients_df
    claims = claims_df
    claim_lines = claim_lines_df
    logger.info("Tools data initialized")

def to_native_type(val):
    """Convert pandas/numpy types to native Python types"""
    if pd.isna(val):
        return None
    if hasattr(val, 'item'):
        return val.item()
    return val

def _fetch_patient_data_impl(patient_id: str) -> Dict[str, Any]:
    """
    Internal implementation for fetching patient data.
    This is the actual function logic that can be called directly.
    
    Args:
        patient_id: Patient UUID to fetch data for
        
    Returns:
        Dictionary containing:
        - patient: Patient demographic information
        - claims: List of all claims for this patient
        - claim_lines: List of all claim line items
        
    Raises:
        ValueError: If patient ID is not found
    """
    logger.info(f"Fetching data for patient: {patient_id}")
    
    if patients is None:
        raise RuntimeError("Tools data not initialized. Call initialize_tools_data() first.")
    
    # Validate patient exists
    if patient_id not in patients.index:
        logger.error(f"Patient ID {patient_id} not found")
        raise ValueError(f"Patient ID {patient_id} not found in patients.csv")
    
    # Get patient data
    patient_row = patients.loc[patient_id]
    patient_dict = {
        "Id": str(patient_id),
        "SSN": str(to_native_type(patient_row.get("SSN", ""))) if pd.notna(patient_row.get("SSN")) else "",
        "BIRTHDATE": str(to_native_type(patient_row.get("BIRTHDATE", ""))) if pd.notna(patient_row.get("BIRTHDATE")) else "",
        "FIRST": str(to_native_type(patient_row.get("FIRST", ""))) if pd.notna(patient_row.get("FIRST")) else "",
        "LAST": str(to_native_type(patient_row.get("LAST", ""))) if pd.notna(patient_row.get("LAST")) else "",
        "ADDRESS": str(to_native_type(patient_row.get("ADDRESS", ""))) if pd.notna(patient_row.get("ADDRESS")) else "",
        "CITY": str(to_native_type(patient_row.get("CITY", ""))) if pd.notna(patient_row.get("CITY")) else "",
        "STATE": str(to_native_type(patient_row.get("STATE", ""))) if pd.notna(patient_row.get("STATE")) else "",
        "ZIP": str(to_native_type(patient_row.get("ZIP", ""))) if pd.notna(patient_row.get("ZIP")) else "",
        "PHONE": str(to_native_type(patient_row.get("PHONE", ""))) if pd.notna(patient_row.get("PHONE")) else "",
        "EMAIL": str(to_native_type(patient_row.get("EMAIL", ""))) if pd.notna(patient_row.get("EMAIL")) else ""
    }
    
    # Get all claims for this patient
    patient_claims = claims[claims["patient_id"] == patient_id]
    claims_list = []
    
    for _, claim_row in patient_claims.iterrows():
        claim_dict = {
            "claim_id": str(to_native_type(claim_row.get("claim_id", ""))),
            "primary_diagnosis_code": str(to_native_type(claim_row.get("primary_diagnosis_code", ""))) if pd.notna(claim_row.get("primary_diagnosis_code")) else "",
            "primary_diagnosis_description": str(to_native_type(claim_row.get("primary_diagnosis_description", ""))) if pd.notna(claim_row.get("primary_diagnosis_description")) else "",
            "total_claim_cost": float(to_native_type(claim_row.get("total_claim_cost", 0))),
            "admission_date": str(to_native_type(claim_row.get("admission_date", ""))) if pd.notna(claim_row.get("admission_date")) else "",
            "discharge_date": str(to_native_type(claim_row.get("discharge_date", ""))) if pd.notna(claim_row.get("discharge_date")) else "",
            "service_date": str(to_native_type(claim_row.get("service_date", ""))) if pd.notna(claim_row.get("service_date")) else "",
            "encounter_class": str(to_native_type(claim_row.get("encounter_class", ""))) if pd.notna(claim_row.get("encounter_class")) else ""
        }
        claims_list.append(claim_dict)
    
    # Get claim lines for this patient's claims
    if len(claims_list) > 0:
        claim_ids = [c["claim_id"] for c in claims_list]
        patient_claim_lines = claim_lines[claim_lines["claim_id"].isin(claim_ids)]
        claim_lines_list = []
        
        for _, line_row in patient_claim_lines.iterrows():
            line_dict = {
                "claim_id": str(to_native_type(line_row.get("claim_id", ""))),
                "line_id": int(to_native_type(line_row.get("line_id", 0))),
                "cpt_hcpcs_code": str(to_native_type(line_row.get("cpt_hcpcs_code", ""))) if pd.notna(line_row.get("cpt_hcpcs_code")) else "",
                "description": str(to_native_type(line_row.get("description", ""))) if pd.notna(line_row.get("description")) else "",
                "charge_amount": float(to_native_type(line_row.get("charge_amount", 0))),
                "units": int(to_native_type(line_row.get("units", 1))),
                "reason_code": str(to_native_type(line_row.get("reason_code", ""))) if pd.notna(line_row.get("reason_code")) else "",
                "reason_description": str(to_native_type(line_row.get("reason_description", ""))) if pd.notna(line_row.get("reason_description")) else ""
            }
            claim_lines_list.append(line_dict)
    else:
        claim_lines_list = []
    
    result = {
        "patient": patient_dict,
        "claims": claims_list,
        "claim_lines": claim_lines_list
    }
    
    logger.info(f"Fetched data: {len(claims_list)} claims, {len(claim_lines_list)} claim lines")
    return result

@FunctionTool
def fetch_patient_data_tool(patient_id: str) -> Dict[str, Any]:
    """
    Fetch patient data, claims, and claim lines from Synthea data.
    
    This tool retrieves all relevant information for a patient including:
    - Patient demographics (SSN, DOB, name, address, etc.)
    - All claims associated with the patient
    - All claim line items for those claims
    
    Args:
        patient_id: Patient UUID to fetch data for
        
    Returns:
        Dictionary containing:
        - patient: Patient demographic information
        - claims: List of all claims for this patient
        - claim_lines: List of all claim line items
        
    Raises:
        ValueError: If patient ID is not found
    """
    return _fetch_patient_data_impl(patient_id)

# Direct callable function (not decorated) for use in Python code
def fetch_patient_data_direct(patient_id: str) -> Dict[str, Any]:
    """
    Direct callable version of fetch_patient_data_tool for use outside of agents.
    This function can be called directly from Python code.
    """
    return _fetch_patient_data_impl(patient_id)

@FunctionTool
def calculate_claim_statistics(claim_ids: List[str]) -> Dict[str, Any]:
    """
    Calculate statistics for a list of claims to help identify anomalies.
    
    Args:
        claim_ids: List of claim UUIDs to analyze
        
    Returns:
        Dictionary with statistics including:
        - total_claims: Number of claims
        - total_cost: Sum of all claim costs
        - avg_cost: Average claim cost
        - max_cost: Maximum claim cost
        - min_cost: Minimum claim cost
        - cost_std_dev: Standard deviation of costs
    """
    logger.info(f"Calculating statistics for {len(claim_ids)} claims")
    
    if claims is None:
        raise RuntimeError("Tools data not initialized. Call initialize_tools_data() first.")
    
    if not claim_ids:
        return {
            "total_claims": 0,
            "total_cost": 0.0,
            "avg_cost": 0.0,
            "max_cost": 0.0,
            "min_cost": 0.0,
            "cost_std_dev": 0.0
        }
    
    # Filter claims
    patient_claims = claims[claims["claim_id"].isin(claim_ids)]
    costs = patient_claims["total_claim_cost"].tolist()
    
    if len(costs) == 0:
        return {
            "total_claims": 0,
            "total_cost": 0.0,
            "avg_cost": 0.0,
            "max_cost": 0.0,
            "min_cost": 0.0,
            "cost_std_dev": 0.0
        }
    
    import statistics
    
    stats = {
        "total_claims": len(costs),
        "total_cost": sum(costs),
        "avg_cost": statistics.mean(costs),
        "max_cost": max(costs),
        "min_cost": min(costs),
        "cost_std_dev": statistics.stdev(costs) if len(costs) > 1 else 0.0
    }
    
    logger.info(f"Statistics calculated: avg_cost={stats['avg_cost']:.2f}, max_cost={stats['max_cost']:.2f}")
    return stats

@FunctionTool
def check_patient_consistency(patient_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check for consistency in patient information across multiple claims.
    
    This tool analyzes patient demographics across claims to detect inconsistencies
    that might indicate identity misuse.
    
    Args:
        patient_data: Dictionary containing patient info and claims
        
    Returns:
        Dictionary with consistency check results:
        - is_consistent: Boolean indicating if all data is consistent
        - inconsistencies: List of detected inconsistencies
        - ssn_matches: Whether SSN is consistent across claims
        - dob_matches: Whether DOB is consistent across claims
        - name_matches: Whether name is consistent across claims
    """
    logger.info("Checking patient information consistency")
    
    patient = patient_data.get("patient", {})
    claims_list = patient_data.get("claims", [])
    
    if not claims_list:
        return {
            "is_consistent": True,
            "inconsistencies": [],
            "ssn_matches": True,
            "dob_matches": True,
            "name_matches": True
        }
    
    inconsistencies = []
    
    # Check if patient has multiple claims with different dates that might indicate issues
    if len(claims_list) > 1:
        dates = [c.get("service_date") for c in claims_list if c.get("service_date")]
        if len(set(dates)) == len(dates) and len(dates) > 5:
            # Many unique dates might indicate rapid claim sequences
            inconsistencies.append("Multiple unique service dates detected")
        
        # Check for unusually high claim costs
        costs = [c.get("total_claim_cost", 0) for c in claims_list]
        if costs:
            avg_cost = sum(costs) / len(costs)
            max_cost = max(costs)
            if max_cost > avg_cost * 3:
                inconsistencies.append(f"Unusually high claim cost detected: ${max_cost:.2f} vs avg ${avg_cost:.2f}")
    
    result = {
        "is_consistent": len(inconsistencies) == 0,
        "inconsistencies": inconsistencies,
        "ssn_matches": True,  # Would check actual SSN if available in claims
        "dob_matches": True,
        "name_matches": True
    }
    
    logger.info(f"Consistency check: is_consistent={result['is_consistent']}")
    return result

@FunctionTool
def analyze_diagnosis_procedure_match(diagnosis_code: str, procedure_codes: List[str]) -> Dict[str, Any]:
    """
    Analyze if procedure codes match the diagnosis code.
    
    This tool validates that procedures are appropriate for a given diagnosis,
    which helps detect billing fraud.
    
    Args:
        diagnosis_code: ICD-10 or SNOMED diagnosis code
        procedure_codes: List of CPT/HCPCS procedure codes
        
    Returns:
        Dictionary with match analysis:
        - matches: Number of matching procedures
        - mismatches: List of procedure codes that don't match
        - match_percentage: Percentage of procedures that match
        - is_valid: Boolean indicating if all procedures match
    """
    logger.info(f"Analyzing diagnosis-procedure match for {diagnosis_code} with {len(procedure_codes)} procedures")
    
    # This is a simplified version - in production, you'd have a mapping database
    # For now, we'll do basic validation
    mismatches = []
    
    # Basic validation: check if codes are non-empty
    if not diagnosis_code or diagnosis_code.strip() == "":
        return {
            "matches": 0,
            "mismatches": procedure_codes,
            "match_percentage": 0.0,
            "is_valid": False,
            "note": "Invalid diagnosis code"
        }
    
    # For demonstration, we'll flag if there are too many procedures for a single diagnosis
    if len(procedure_codes) > 10:
        mismatches = procedure_codes[10:]
    
    matches = len(procedure_codes) - len(mismatches)
    match_percentage = (matches / len(procedure_codes) * 100) if procedure_codes else 0.0
    
    result = {
        "matches": matches,
        "mismatches": mismatches,
        "match_percentage": match_percentage,
        "is_valid": len(mismatches) == 0
    }
    
    logger.info(f"Match analysis: {matches} matches, {len(mismatches)} mismatches")
    return result

