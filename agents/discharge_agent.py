import pandas as pd
from pydantic import BaseModel

class DischargeOutput(BaseModel):
    patient_id: str
    discharge_ready_flag: bool
    blockers: list
    delay_hours: float
    priority_level: str
    validation: dict

def run_discharge_agent(patient_id, patients_csv="patients.csv"):

    try:
        df = pd.read_csv(patients_csv)
    except Exception as e:
        return {
            "patient_id": patient_id,
            "error": "CSV file not loaded",
            "details": str(e)
        }

    patient = df[df["patient_id"] == patient_id]

    if patient.empty:
        return {
            "patient_id": patient_id,
            "discharge_ready_flag": False,
            "blockers": ["patient_not_found"],
            "delay_hours": 0,
            "priority_level": "HIGH",
            "validation": {"json_valid": False, "missing": ["patient_id"]}
        }

    task_status = patient["task"].iloc[0]
    blockers = []
    delay_hours = 0

    if task_status == "Pending Lab":
        blockers.append("pending_labs")
        delay_hours += 3

    if task_status == "Pending Imaging":
        blockers.append("pending_imaging")
        delay_hours += 4

    if task_status == "Missing Consult":
        blockers.append("missing_consultation")
        delay_hours += 2

    if task_status == "None":
        return {
            "patient_id": patient_id,
            "discharge_ready_flag": True,
            "blockers": [],
            "delay_hours": 0,
            "priority_level": "LOW",
            "validation": {"json_valid": True, "missing": []}
        }

    priority = "HIGH" if delay_hours > 5 else "MEDIUM"

    return {
        "patient_id": patient_id,
        "discharge_ready_flag": False,
        "blockers": blockers,
        "delay_hours": delay_hours,
        "priority_level": priority,
        "validation": {"json_valid": True, "missing": []}
    }
