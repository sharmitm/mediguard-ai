# Backend Documentation

This document explains all functions and components in the MediGuard AI backend system, now using Google ADK instead of LangChain/LangGraph.

## Table of Contents

- [Main Workflow (`main.py`)](#main-workflow-mainpy)
- [Tools (`tools_adk.py`)](#tools-tools_adkpy)
- [API Server (`api_server.py`)](#api-server-api_serverpy)
- [Data Loading](#data-loading)
- [Agent Functions](#agent-functions)
- [Observability](#observability)

---

## Main Workflow (`main.py`)

### `load_all_data()`

**Purpose:** Loads all Synthea CSV files into pandas DataFrames at application startup.

**What it does:**
- Reads `data1/patients.csv` and indexes by patient `Id` (UUID)
- Reads `data1/claims.csv` with all claim records
- Reads `data1/claim_lines.csv` with detailed billing line items
- Returns three DataFrames for efficient querying

**Returns:**
- `patients_df` - DataFrame indexed by patient UUID
- `claims_df` - DataFrame with all claims
- `claim_lines_df` - DataFrame with all claim line items

**Usage:** Called once at module import to load data into memory.

---

### `fetch_patient_data(patient_id: str)`

**Purpose:** Retrieves all data for a specific patient from the loaded DataFrames.

**Parameters:**
- `patient_id` (str): Patient UUID to fetch data for

**What it does:**
1. Validates that the patient exists in the patients DataFrame
2. Extracts patient demographics (SSN, DOB, name, address, etc.)
3. Filters claims DataFrame to get all claims for this patient
4. Filters claim_lines DataFrame to get all billing lines for those claims
5. Converts numpy/pandas types to native Python types for JSON serialization
6. Returns structured dictionary with patient, claims, and claim_lines

**Returns:**
```python
{
    "patient": {
        "Id": "uuid",
        "SSN": "999-33-6229",
        "BIRTHDATE": "2000-10-06",
        "FIRST": "John",
        "LAST": "Doe",
        # ... other fields
    },
    "claims": [
        {
            "claim_id": "uuid",
            "primary_diagnosis_code": "I10",
            "total_claim_cost": 704.20,
            # ... other fields
        }
    ],
    "claim_lines": [
        {
            "claim_id": "uuid",
            "cpt_hcpcs_code": "99214",
            "charge_amount": 215.70,
            # ... other fields
        }
    ],
    "tasks": []  # Empty list (for Agent 3)
}
```

**Raises:** `ValueError` if patient ID not found

---

### `identity_agent` (ADK LlmAgent)

**Purpose:** Agent 1 - Analyzes patient data for identity fraud and suspicious claims using Google ADK.

**What it does:**
1. Uses ADK `LlmAgent` with custom tools
2. Agent calls `fetch_patient_data_tool` to retrieve patient data
3. Agent can use `calculate_claim_statistics` and `check_patient_consistency` tools
4. LLM analyzes for:
   - Duplicate/inconsistent patient information across claims
   - Suspicious diagnosis-procedure combinations
   - Unusually high claim amounts
   - Identity misuse patterns
5. Returns JSON with required fields: `fraud_risk_score`, `identity_misuse_flag`, `reasons`

**Tools Available:**
- `fetch_patient_data_tool` - Retrieves patient, claims, and claim_lines
- `calculate_claim_statistics` - Calculates statistical metrics
- `check_patient_consistency` - Checks data consistency

**Output Format:**
```python
{
    "fraud_risk_score": 45,
    "identity_misuse_flag": true,
    "reasons": ["Duplicate patient information", "Suspicious pattern"]
}
```

**Raises:** `ValueError` if LLM output doesn't match required structure

---

### `billing_agent` (ADK LlmAgent)

**Purpose:** Agent 2 - Analyzes billing for fraud based on identity analysis results using Google ADK.

**What it does:**
1. Uses ADK `LlmAgent` with billing-specific tools
2. Receives identity analysis from previous agent
3. Agent can use `calculate_claim_statistics` and `analyze_diagnosis_procedure_match` tools
4. LLM checks for:
   - Procedures not supported by diagnosis
   - Duplicate/add-on procedures
   - Charges above normal ranges
   - Suspicious billing combinations
5. Returns JSON with billing analysis results

**Tools Available:**
- `calculate_claim_statistics` - Analyzes claim costs
- `analyze_diagnosis_procedure_match` - Validates procedure-diagnosis matches

**Output Format:**
```python
{
    "billing_risk_score": 25,
    "billing_flags": ["normal_range"],
    "billing_explanation": "No billing anomalies"
}
```

---

### `discharge_agent` (ADK LlmAgent)

**Purpose:** Agent 3 - Assesses discharge readiness and identifies blockers using Google ADK.

**What it does:**
1. Uses ADK `LlmAgent` with discharge assessment tools
2. Receives identity and billing analysis results
3. Uses tools to:
   - Check for active inpatient encounters (`get_active_encounters`)
   - Find pending procedures (`get_pending_procedures`)
   - Check fraud-related discharge blockers (`check_discharge_readiness`)
4. LLM determines:
   - If patient is ready for discharge
   - What blockers exist (pending labs, scans, consults, fraud investigations)
   - Estimated delay hours based on blocker types
5. Returns JSON with discharge assessment

**Tools Available:**
- `get_active_encounters` - Checks for active inpatient encounters
- `get_pending_procedures` - Finds pending labs, scans, consults
- `check_discharge_readiness` - Checks if fraud blocks discharge
- `fetch_patient_data_tool` - Retrieves patient data

**Output Format:**
```python
{
    "discharge_ready": false,
    "blockers": ["Pending Lab", "Missing Consult"],
    "delay_hours": 24
}
```

---

### `analyze_patient(patient_id: str)`

**Purpose:** Main entry point to run all three agents in sequence using ADK SequentialAgent.

**What it does:**
1. Uses ADK `SequentialAgent` to run agents in sequence
2. Workflow runs: identity_agent → billing_agent → discharge_agent
3. Each agent runs with its own session
4. Combines results from all agents
5. Returns the complete workflow state with all agent results

**Parameters:**
- `patient_id` (str): Patient UUID to analyze

**Returns:** Complete workflow state dictionary with identity, billing, discharge, and final results

**Usage:**
```python
result = analyze_patient("341de73b-56e5-6f58-c32f-9d56a1290e2f")
print(result["final"])  # Combined results
```

**Note:** ADK SequentialAgent automatically handles agent sequencing and state passing.

---

### `analyze_agent1_only(patient_id: str)`

**Purpose:** Runs only Agent 1 (Identity & Claims Fraud) for testing using ADK.

**What it does:**
1. Uses ADK `identity_agent.run()` directly
2. Agent automatically calls `fetch_patient_data_tool` when needed
3. Validates output structure
4. Returns Agent 1 results only

**Parameters:**
- `patient_id` (str): Patient UUID

**Returns:** Agent 1 results dictionary with fraud_risk_score, identity_misuse_flag, reasons

**Usage:** Useful for testing Agent 1 independently

**Example:**
```python
result = analyze_agent1_only("341de73b-56e5-6f58-c32f-9d56a1290e2f")
# Returns: {"fraud_risk_score": 45, "identity_misuse_flag": true, "reasons": [...]}
```

---

### `get_sample_patient_ids(limit: int = 10)`

**Purpose:** Gets a list of sample patient IDs for testing.

**What it does:**
- Returns first N patient IDs from the loaded patients DataFrame

**Parameters:**
- `limit` (int): Number of patient IDs to return (default: 10)

**Returns:** List of patient UUID strings

---

## API Server (`api_server.py`)

### `get_sample_ids(limit: int = 10)`

**Purpose:** API endpoint to get sample patient IDs.

**Endpoint:** `GET /api/sample-ids?limit=10`

**What it does:**
- Calls `get_sample_patient_ids()` from main.py
- Returns JSON with list of patient IDs

**Response:**
```json
{
    "ids": ["uuid1", "uuid2", ...]
}
```

---

### `analyze_patient_endpoint(request: PatientAnalysisRequest)`

**Purpose:** API endpoint to run full analysis through all agents.

**Endpoint:** `POST /api/analyze`

**What it does:**
1. Validates patient exists
2. Calls `analyze_patient()` to run all agents
3. Structures response to match frontend expectations
4. Separates identity, billing, discharge, and final results

**Request Body:**
```json
{
    "patient_id": "uuid-here"
}
```

**Response:**
```json
{
    "identity": {
        "fraud_risk_score": 45,
        "identity_misuse_flag": true,
        "reasons": [...]
    },
    "billing": {
        "billing_fraud_score": 25,
        "billing_flags": [...],
        "billing_explanation": "..."
    },
    "discharge": {
        "discharge_ready": false,
        "blockers": [...],
        "delay_hours": 24
    },
    "final": {
        // Combined results
    }
}
```

**Error Responses:**
- `404` - Patient not found
- `500` - Analysis error (includes traceback)

---

### `analyze_agent1_only_endpoint(request: PatientAnalysisRequest)`

**Purpose:** API endpoint to run only Agent 1.

**Endpoint:** `POST /api/analyze/agent1`

**What it does:**
- Calls `analyze_agent1_only()` from main.py
- Returns only Agent 1 results

**Use Case:** Testing or when only identity fraud detection is needed

---

## Data Loading

### CSV Files Structure

**`data1/patients.csv`:**
- Indexed by `Id` (UUID)
- Contains: SSN, BIRTHDATE, FIRST, LAST, ADDRESS, CITY, STATE, ZIP, PHONE, EMAIL

**`data1/claims.csv`:**
- Contains: `claim_id`, `patient_id`, `primary_diagnosis_code`, `total_claim_cost`, dates
- Linked to patients via `patient_id`

**`data1/claim_lines.csv`:**
- Contains: `claim_id`, `line_id`, `cpt_hcpcs_code`, `charge_amount`, `units`
- Linked to claims via `claim_id`

### Data Flow

1. **Startup:** `load_all_data()` loads all CSVs into DataFrames
2. **Request:** `fetch_patient_data()` filters DataFrames for specific patient
3. **Analysis:** Agents receive structured data dictionaries
4. **Response:** Results are JSON-serialized and returned

---

## Agent Prompts

All agents use structured prompts that:
- Instruct LLM to return ONLY valid JSON
- Specify exact output field names and types
- Provide examples of expected output
- Include analysis instructions

**Prompt Structure:**
- System message: Role definition and JSON-only requirement
- Human message: Data to analyze + output format specification + example

---

## Error Handling

**Common Errors:**

1. **Patient Not Found:**
   - `ValueError` raised by `fetch_patient_data()`
   - Caught and returned as 404 HTTP error

2. **Invalid LLM Response:**
   - JSON parsing errors caught
   - Markdown code blocks stripped automatically
   - Validation ensures required fields exist

3. **Type Errors:**
   - Numpy types converted to native Python types
   - NaN values handled gracefully

---

## Workflow Architecture

The system uses **Google ADK SequentialAgent** to orchestrate the agent workflow:

```
Start → identity_agent → billing_agent → discharge_agent → End
```

Each agent:
- Is an ADK `LlmAgent` with its own tools and system instructions
- Uses `InMemorySessionService` for session management
- Can call FunctionTools to retrieve and analyze data
- Returns structured JSON responses
- Results are combined in the final workflow output

ADK SequentialAgent automatically:
- Runs agents in sequence
- Manages session state
- Handles tool calls
- Provides built-in observability

---

## Tools (`tools_adk.py`)

The system includes custom ADK FunctionTools that agents can call:

### `fetch_patient_data_tool(patient_id: str)`

**Purpose:** Retrieves all patient data, claims, and claim lines.

**Returns:** Dictionary with patient demographics, claims list, and claim_lines list.

**Used by:** Identity agent to get patient data for analysis.

### `calculate_claim_statistics(claim_ids: List[str])`

**Purpose:** Calculates statistical metrics for claims (total_cost, avg_cost, max_cost, etc.).

**Returns:** Dictionary with statistics including total_claims, total_cost, avg_cost, max_cost, min_cost, cost_std_dev.

**Used by:** Identity and Billing agents to identify anomalies.

### `check_patient_consistency(patient_data: Dict[str, Any])`

**Purpose:** Checks for consistency in patient information across claims.

**Returns:** Dictionary with consistency flags and detected inconsistencies.

**Used by:** Identity agent to detect identity misuse patterns.

### `analyze_diagnosis_procedure_match(diagnosis_code: str, procedure_codes: List[str])`

**Purpose:** Validates that procedure codes match the diagnosis code.

**Returns:** Dictionary with match analysis including matches, mismatches, match_percentage, is_valid.

**Used by:** Billing agent to detect billing fraud.

---

## Observability

The system uses ADK's built-in observability features:

### Logging

- **Setup:** `setup_logging(level=logging.INFO)` at module initialization
- **Structured Logging:** JSON-formatted logs with event types, timestamps, durations
- **Log Points:**
  - Agent start/completion with duration metrics
  - Tool calls with parameters
  - Errors with stack traces
  - API requests/responses

### Metrics

- Agent execution times
- Tool call frequencies
- Success/failure rates
- API response times

### Session Management

- Uses `InMemorySessionService` for session state
- Each agent run uses a unique session_id
- Session data persists during workflow execution

