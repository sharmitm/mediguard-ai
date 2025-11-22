# MediGuard AI Agent

Healthcare fraud detection and discharge management system using LangGraph and Gemini 2.5.

## Features
- Identity & Claims Fraud Detection
- Billing Fraud Analysis
- Discharge Readiness Assessment

##  Agent 3 – **Discharge Readiness Agent**

### **Purpose**
The Discharge Readiness Agent assesses a patient's pending responsibilities and looks for anything that could cause a delay in discharge.  
 It employs LLM reasoning to comprehend task descriptions, identify bottlenecks, and predict the projected delay.

### **What It Checks**
- Pending lab tests
- Pending imaging or scans
- Missing specialist consultations
- Incomplete or pending tasks
- Any discharge-blocking instructions in the patient case

### **Input**
The agent receives:
- Patient’s task list  
(from the synthetic data in `patients.csv`)

### **Output (JSON)**
The agent returns structured JSON:
```json
{
  "discharge_ready": true/false,
  "blockers": ["list of reasons"],
  "delay_hours": <number>,
  "priority_level": "LOW" | "MEDIUM" | "HIGH"
}



To identify clinical and administrative blockers that prevent a patient from being safely discharged.
## Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and add your Google API key
4. Run: `python main.py`

## Requirements
- Python 3.9+
- Google Gemini API key
