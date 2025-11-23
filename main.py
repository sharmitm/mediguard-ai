"""
MediGuard AI - Healthcare Fraud Detection System
Implemented using Google ADK instead of LangChain/LangGraph
"""
import json
import os
import sys
import logging
import time
import asyncio
import uuid
from typing import Dict, Any
import pandas as pd
from dotenv import load_dotenv

# Google ADK imports
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.models import Gemini
from google.adk.sessions import InMemorySessionService
from google.adk.runners import types

# Import tools
from tools_adk import (
    fetch_patient_data_tool,
    fetch_patient_data_direct,
    calculate_claim_statistics,
    check_patient_consistency,
    analyze_diagnosis_procedure_match,
    get_active_encounters,
    get_pending_procedures,
    check_discharge_readiness,
    initialize_tools_data
)

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load all Synthea data from data1/ folder
def load_all_data():
    """Load all Synthea CSV files from data1/ folder"""
    data_dir = "data1"
    patients_df = pd.read_csv(f"{data_dir}/patients.csv").set_index("Id")
    claims_df = pd.read_csv(f"{data_dir}/claims.csv")
    claim_lines_df = pd.read_csv(f"{data_dir}/claim_lines.csv")
    
    # Load encounters and procedures if available
    encounters_df = None
    procedures_df = None
    try:
        encounters_df = pd.read_csv(f"{data_dir}/encounters.csv")
        logger.info(f"Loaded encounters: {len(encounters_df)} records")
    except FileNotFoundError:
        logger.warning("encounters.csv not found, will use claims data for encounters")
    except Exception as e:
        logger.warning(f"Error loading encounters.csv: {e}")
    
    try:
        procedures_df = pd.read_csv(f"{data_dir}/procedures.csv")
        logger.info(f"Loaded procedures: {len(procedures_df)} records")
    except FileNotFoundError:
        logger.warning("procedures.csv not found")
    except Exception as e:
        logger.warning(f"Error loading procedures.csv: {e}")
    
    return patients_df, claims_df, claim_lines_df, encounters_df, procedures_df

# Load data at startup
patients, claims, claim_lines, encounters, procedures = load_all_data()
logger.info(f"Loaded data: {len(patients)} patients, {len(claims)} claims, {len(claim_lines)} claim lines")

# Initialize tools with data
initialize_tools_data(patients, claims, claim_lines, encounters, procedures)

# Initialize Gemini model
model = Gemini(
    model_name="gemini-2.5-flash-lite",
    temperature=0.0,
    api_key=os.getenv("GOOGLE_API_KEY")
)

# ============================================================================
# AGENT DEFINITIONS
# ============================================================================

# Agent 1: Identity & Claims Fraud Detection
identity_agent = LlmAgent(
    name="identity_agent",
    model=model,
    tools=[
        fetch_patient_data_tool,
        calculate_claim_statistics,
        check_patient_consistency
    ],
    instruction="""You are MediGuard Identity & Claims Fraud Detection Agent. 

Your role is to analyze patient data for fraud and identity misuse patterns.

You MUST:
1. Use the fetch_patient_data_tool to get patient information when given a patient_id
2. Analyze the data for:
   - Duplicate or inconsistent patient information across claims (compare SSN, DOB, name, address)
   - Suspicious diagnosis-procedure combinations (procedures that don't match diagnoses)
   - Claims with unusually high or unrealistic amounts (compare total_claim_cost to typical ranges)
   - Patterns commonly associated with identity misuse (multiple claims with different patient details, rapid claim sequences, etc.)
3. Use calculate_claim_statistics tool to get statistical insights
4. Use check_patient_consistency tool to verify data consistency
5. AFTER using the tools, you MUST provide your analysis result as text

You MUST respond with ONLY valid JSON in this exact format:
{
    "fraud_risk_score": <number 0-100>,
    "identity_misuse_flag": <boolean>,
    "reasons": [<array of strings>]
}

IMPORTANT: After calling any tools, you MUST provide your final answer as text. Do NOT stop after calling tools - always provide the JSON response.

Do NOT include markdown code blocks, explanations, or any text outside the JSON object.
Return ONLY the raw JSON object.

Example output:
{"fraud_risk_score": 45, "identity_misuse_flag": true, "reasons": ["Duplicate patient information across multiple claims", "Suspicious diagnosis-procedure combination detected"]}"""
)

# Agent 2: Billing Fraud Analysis
billing_agent = LlmAgent(
    name="billing_agent",
    model=model,
    tools=[
        calculate_claim_statistics,
        analyze_diagnosis_procedure_match
    ],
    instruction="""You are MediGuard Billing Fraud Agent.

Your role is to analyze billing for fraud based on identity analysis results.

You MUST:
1. Review the identity analysis results provided
2. Use calculate_claim_statistics tool to analyze claim costs
3. Use analyze_diagnosis_procedure_match tool to verify procedure-diagnosis matches
4. Check for:
   - Procedures not supported by diagnosis
   - Duplicate/add-on procedures
   - Charges above normal ranges
   - Suspicious billing combinations

You MUST respond with ONLY valid JSON in this exact format:
{
    "billing_risk_score": <number 0-100>,
    "billing_flags": [<array of strings>],
    "billing_explanation": <string>
}

Do NOT include markdown code blocks, explanations, or any text outside the JSON object.
Return ONLY the raw JSON object.

Example output:
{"billing_risk_score": 15, "billing_flags": ["normal_range"], "billing_explanation": "No billing anomalies"}"""
)

# Agent 3: Discharge Blockers Assessment
discharge_agent = LlmAgent(
    name="discharge_agent",
    model=model,
    tools=[
        get_active_encounters,
        get_pending_procedures,
        check_discharge_readiness,
        fetch_patient_data_tool
    ],
    instruction="""You are MediGuard Discharge Agent.

Your role is to assess discharge readiness and identify blockers that prevent patient discharge.

You MUST:
1. Use get_active_encounters tool to check if patient has an active inpatient encounter
2. If patient has active encounter, use get_pending_procedures tool to find pending procedures
3. Use check_discharge_readiness tool to check if fraud analysis blocks discharge
4. Analyze all blockers:
   - Pending labs (blood tests, cultures, etc.)
   - Pending scans (imaging studies, X-rays, CT scans, etc.)
   - Pending consults (specialist consultations, referrals)
   - Fraud-related blockers (if identity/billing fraud detected)
5. Determine if patient is ready for discharge
6. Estimate delay hours based on blockers:
   - Labs: 4-8 hours
   - Scans: 2-6 hours
   - Consults: 4-24 hours
   - Fraud investigation: 24-48 hours
   - Multiple blockers: sum of individual delays

CRITICAL INSTRUCTIONS: 
- If patient has no active encounter, they are ready for discharge
- If fraud blocker exists, discharge is NOT ready
- If no blockers exist, patient is ready for discharge
- AFTER calling ANY tools, you MUST ALWAYS provide your final answer as TEXT. 
- Do NOT stop after calling tools - you MUST provide the JSON response as text.
- Even if tools return all the information you need, you MUST still output your analysis as text JSON.
- The workflow requires a text response - function calls alone are not sufficient.

You MUST respond with ONLY valid JSON in this exact format:
{
    "discharge_ready": <boolean>,
    "blockers": [<array of strings describing blockers>],
    "delay_hours": <number - estimated hours until discharge ready>
}

Do NOT include markdown code blocks, explanations, or any text outside the JSON object.
Return ONLY the raw JSON object.

Example outputs:
{"discharge_ready": true, "blockers": [], "delay_hours": 0}
{"discharge_ready": false, "blockers": ["Pending blood lab results", "CT scan scheduled"], "delay_hours": 8}
{"discharge_ready": false, "blockers": ["Identity fraud investigation required"], "delay_hours": 36}"""
)

# Create Sequential workflow (replaces StateGraph)
workflow = SequentialAgent(
    name="mediguard_workflow",
    sub_agents=[identity_agent, billing_agent, discharge_agent]
)

# Create Runner for executing agents
from google.adk import Runner
session_service = InMemorySessionService()
runner = Runner(
    agent=workflow,
    app_name="mediguard_workflow",
    session_service=session_service
)
identity_runner = Runner(
    agent=identity_agent,
    app_name="identity_agent",
    session_service=session_service
)
discharge_runner = Runner(
    agent=discharge_agent,
    app_name="discharge_agent",
    session_service=session_service
)

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def parse_agent_response(response: Any) -> str:
    """Parse ADK agent response to extract text content from events"""
    # If response is a list of events, find the final response
    if isinstance(response, list):
        # Find the final response event
        final_event = None
        for event in response:
            if hasattr(event, 'is_final_response') and event.is_final_response():
                final_event = event
                break
        # If no final event found, use the last event
        if final_event is None and response:
            final_event = response[-1]
        response = final_event
    
    # Extract content from event
    if response and hasattr(response, 'content'):
        content = response.content
        
        # Handle different content types
        if content is None:
            logger.warning(f"Response content is None for {type(response)}")
            return ""
        
        # Content is a list of Content objects, extract text from parts
        if isinstance(content, list) and len(content) > 0:
            # Look through all content items, not just the last one
            # The model response might be in a different content item
            for content_item in reversed(content):  # Start from end
                if content_item is None:
                    continue
                if hasattr(content_item, 'parts') and content_item.parts:
                    # Extract text from all parts (skip function_call parts)
                    text_parts = []
                    for part in content_item.parts:
                        # Check if part has text attribute and it's not None/empty
                        if hasattr(part, 'text') and part.text:
                            text_parts.append(str(part.text).strip())
                    if text_parts:
                        return " ".join(text_parts)
        elif content and not isinstance(content, list):
            # Single content object
            if hasattr(content, 'parts') and content.parts:
                text_parts = []
                for part in content.parts:
                    if hasattr(part, 'text') and part.text:
                        text_parts.append(str(part.text).strip())
                if text_parts:
                    return " ".join(text_parts)
    
    # Try to get text directly from response if it's a string
    if isinstance(response, str):
        return response
    
    # Try to access text attribute
    if hasattr(response, 'text') and response.text:
        return str(response.text)
    
    # Debug: log what we got
    logger.warning(f"Could not extract text from response: {type(response)}")
    if response:
        logger.warning(f"Response attributes: {[x for x in dir(response) if not x.startswith('_')][:15]}")
        if hasattr(response, 'content'):
            logger.warning(f"Content type: {type(response.content)}, value: {response.content}")
    
    # Don't return str(response) - that's the object representation, not the actual text
    return ""

def clean_json_response(content: str) -> str:
    """Clean JSON response by removing markdown code blocks"""
    content = content.strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()
    return content

def validate_identity_result(result: Dict[str, Any]) -> None:
    """Validate Agent 1 (Identity) output structure"""
    required_fields = ["fraud_risk_score", "identity_misuse_flag", "reasons"]
    for field in required_fields:
        if field not in result:
            raise ValueError(f"Agent 1 output missing required field: {field}")
    
    if not isinstance(result["fraud_risk_score"], (int, float)):
        raise ValueError("fraud_risk_score must be a number")
    if not isinstance(result["identity_misuse_flag"], bool):
        raise ValueError("identity_misuse_flag must be a boolean")
    if not isinstance(result["reasons"], list):
        raise ValueError("reasons must be an array")

def validate_billing_result(result: Dict[str, Any]) -> None:
    """Validate Agent 2 (Billing) output structure"""
    required_fields = ["billing_risk_score", "billing_flags", "billing_explanation"]
    for field in required_fields:
        if field not in result:
            raise ValueError(f"Agent 2 output missing required field: {field}")

def validate_discharge_result(result: Dict[str, Any]) -> None:
    """Validate Agent 3 (Discharge) output structure"""
    required_fields = ["discharge_ready", "blockers", "delay_hours"]
    for field in required_fields:
        if field not in result:
            raise ValueError(f"Agent 3 output missing required field: {field}")

def analyze_patient(patient_id: str) -> Dict[str, Any]:
    """
    Main function to analyze a patient through all agents using ADK SequentialAgent.
    
    Args:
        patient_id: Patient UUID to analyze
        
    Returns:
        Dictionary with complete workflow state including identity, billing, discharge, and final results
    """
    try:
        return asyncio.run(analyze_patient_async(patient_id))
    finally:
        # Clean up any pending async tasks
        try:
            loop = asyncio.get_event_loop()
            if not loop.is_closed():
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
        except:
            pass

async def analyze_patient_async(patient_id: str) -> Dict[str, Any]:
    """
    Async implementation of analyze_patient.
    """
    start_time = time.time()
    logger.info(f"Starting full workflow analysis for patient: {patient_id}")
    
    try:
        # Run agents individually in sequence (faster than SequentialAgent + individual runs)
        # This avoids running agents twice and gives us all intermediate results
        
        # Step 1: Run identity agent
        logger.info(f"Step 1: Running identity agent for patient {patient_id}")
        
        # Use unique session IDs to avoid conflicts between requests
        session_suffix = str(uuid.uuid4())[:8]  # Short unique ID
        identity_session_id = f"session_{patient_id}_identity_{session_suffix}"
        
        identity_session = session_service.create_session_sync(
            app_name="identity_agent",
            user_id=f"user_{patient_id}",
            session_id=identity_session_id
        )
        
        # Run identity agent separately to get its result
        identity_prompt = f"Analyze patient {patient_id} for identity fraud. Use fetch_patient_data_tool to get data."
        identity_message = types.Content(
            parts=[types.Part(text=identity_prompt)],
            role="user"
        )
        identity_events = []
        async for event in identity_runner.run_async(
            user_id=f"user_{patient_id}",
            session_id=identity_session_id,
            new_message=identity_message
        ):
            identity_events.append(event)
            # Log event details for debugging
            event_type = event.__class__.__name__ if hasattr(event, '__class__') else str(type(event))
            logger.debug(f"Identity event {len(identity_events)}: {event_type}")
            if hasattr(event, 'content') and event.content:
                logger.debug(f"  Event has content: {type(event.content)}")
        
        logger.info(f"Identity agent completed with {len(identity_events)} events")
        
        # If last event has None content, agent may need a follow-up to generate response
        # Check if we have events and the last one has None or empty content
        needs_followup = False
        if identity_events:
            last_event = identity_events[-1]
            if hasattr(last_event, 'content'):
                if last_event.content is None:
                    needs_followup = True
                elif isinstance(last_event.content, list) and len(last_event.content) == 0:
                    needs_followup = True
        
        if needs_followup:
            logger.warning("Last event has None content - sending follow-up to get JSON response")
            followup_message = types.Content(
                parts=[types.Part(text="Please provide your analysis result as JSON. Return ONLY the JSON object with fraud_risk_score, identity_misuse_flag, and reasons fields. Do not include markdown or explanations.")],
                role="user"
            )
            followup_events = []
            async for event in identity_runner.run_async(
                user_id=f"user_{patient_id}",
                session_id=identity_session_id,
                new_message=followup_message
            ):
                followup_events.append(event)
                logger.debug(f"Followup identity event {len(followup_events)}: {type(event).__name__}")
            
            # Add followup events to the main list
            identity_events.extend(followup_events)
            logger.info(f"Added {len(followup_events)} followup events. Total events: {len(identity_events)}")
        
        # Try to extract response from events, prioritizing model response events
        identity_content = None
        all_text = []
        
        # First, try to find a model response event (after function calls)
        # Iterate in reverse to get the most recent response first
        for i, event in enumerate(reversed(identity_events)):
            if hasattr(event, 'content') and event.content:
                content_list = event.content if isinstance(event.content, list) else [event.content]
                for content_item in content_list:
                    if content_item and hasattr(content_item, 'parts') and content_item.parts:
                        for part in content_item.parts:
                            # Look for text parts (skip function_call parts)
                            if hasattr(part, 'text') and part.text:
                                text = str(part.text).strip()
                                if text:
                                    logger.debug(f"Found text in event {len(identity_events) - i - 1}: {text[:100]}...")
                                    all_text.append(text)
                                    if not identity_content:  # Use first non-empty text found
                                        identity_content = text
        
        # If we found text, join it all
        if all_text:
            identity_content = " ".join(all_text)
            logger.info(f"Extracted identity content from events: {len(identity_content)} chars")
        elif identity_events:
            # If no text found, log detailed event information for debugging
            logger.error(f"Identity agent returned no text content. Event count: {len(identity_events)}")
            for i, event in enumerate(identity_events):
                event_type = type(event).__name__ if hasattr(type(event), '__name__') else str(type(event))
                logger.error(f"Event {i}: type={event_type}, has content={hasattr(event, 'content')}")
                if hasattr(event, 'content'):
                    logger.error(f"  Content type: {type(event.content)}, value: {event.content}")
                    if event.content and isinstance(event.content, list):
                        for j, content_item in enumerate(event.content):
                            logger.error(f"    Content item {j}: {type(content_item)}")
                            if content_item and hasattr(content_item, 'parts'):
                                logger.error(f"      Parts: {content_item.parts}")
            raise ValueError(f"Identity agent returned empty response. Events: {len(identity_events)}")
        else:
            raise ValueError("Identity agent returned no events")
        identity_content = clean_json_response(identity_content)
        logger.info(f"Identity agent response length: {len(identity_content)} chars, preview: {identity_content[:200]}")
        identity_result = json.loads(identity_content)
        validate_identity_result(identity_result)
        logger.info(f"Step 1 complete: Identity agent fraud_score={identity_result.get('fraud_risk_score', 0)}")
        
        # Step 2: Run billing agent with identity result
        logger.info(f"Step 2: Running billing agent for patient {patient_id}")
        # Create session for billing agent with unique ID
        billing_session_id = f"session_{patient_id}_billing_{session_suffix}"
        billing_session = session_service.create_session_sync(
            app_name="billing_agent",
            user_id=f"user_{patient_id}",
            session_id=billing_session_id
        )
        
        # Run billing agent with identity result
        billing_runner = Runner(agent=billing_agent, app_name="billing_agent", session_service=session_service)
        billing_prompt = f"Analyze billing fraud based on this identity analysis: {json.dumps(identity_result)}"
        billing_message = types.Content(
            parts=[types.Part(text=billing_prompt)],
            role="user"
        )
        billing_events = []
        async for event in billing_runner.run_async(
            user_id=f"user_{patient_id}",
            session_id=billing_session_id,
            new_message=billing_message
        ):
            billing_events.append(event)
            # Log event details for debugging
            event_type = event.__class__.__name__ if hasattr(event, '__class__') else str(type(event))
            logger.debug(f"Billing event {len(billing_events)}: {event_type}")
            if hasattr(event, 'content') and event.content:
                logger.debug(f"  Event has content: {type(event.content)}")
        
        logger.info(f"Billing agent completed with {len(billing_events)} events")
        
        # If last event has None content, agent may need a follow-up to generate response
        needs_followup = False
        if billing_events:
            last_event = billing_events[-1]
            if hasattr(last_event, 'content'):
                if last_event.content is None:
                    needs_followup = True
                elif isinstance(last_event.content, list) and len(last_event.content) == 0:
                    needs_followup = True
        
        if needs_followup:
            logger.warning("Last billing event has None content - sending follow-up to get JSON response")
            followup_message = types.Content(
                parts=[types.Part(text="Please provide your analysis result as JSON. Return ONLY the JSON object with billing_risk_score, billing_flags, and billing_explanation fields. Do not include markdown or explanations.")],
                role="user"
            )
            followup_events = []
            async for event in billing_runner.run_async(
                user_id=f"user_{patient_id}",
                session_id=billing_session_id,
                new_message=followup_message
            ):
                followup_events.append(event)
                logger.debug(f"Followup billing event {len(followup_events)}: {type(event).__name__}")
            
            # Add followup events to the main list
            billing_events.extend(followup_events)
            logger.info(f"Added {len(followup_events)} followup events. Total events: {len(billing_events)}")
        
        # Try to extract response from events, prioritizing model response events
        billing_content = None
        all_text = []
        
        # First, try to find a model response event (after function calls)
        # Iterate in reverse to get the most recent response first
        for i, event in enumerate(reversed(billing_events)):
            if hasattr(event, 'content') and event.content:
                content_list = event.content if isinstance(event.content, list) else [event.content]
                for content_item in content_list:
                    if content_item and hasattr(content_item, 'parts') and content_item.parts:
                        for part in content_item.parts:
                            # Look for text parts (skip function_call parts)
                            if hasattr(part, 'text') and part.text:
                                text = str(part.text).strip()
                                if text:
                                    logger.debug(f"Found text in billing event {len(billing_events) - i - 1}: {text[:100]}...")
                                    all_text.append(text)
                                    if not billing_content:  # Use first non-empty text found
                                        billing_content = text
        
        # If we found text, join it all
        if all_text:
            billing_content = " ".join(all_text)
            logger.info(f"Extracted billing content from events: {len(billing_content)} chars")
        elif billing_events:
            # If no text found, log detailed event information for debugging
            logger.error(f"Billing agent returned no text content. Event count: {len(billing_events)}")
            for i, event in enumerate(billing_events):
                event_type = type(event).__name__ if hasattr(type(event), '__name__') else str(type(event))
                logger.error(f"Event {i}: type={event_type}, has content={hasattr(event, 'content')}")
                if hasattr(event, 'content'):
                    logger.error(f"  Content type: {type(event.content)}, value: {event.content}")
                    if event.content and isinstance(event.content, list):
                        for j, content_item in enumerate(event.content):
                            logger.error(f"    Content item {j}: {type(content_item)}")
                            if content_item and hasattr(content_item, 'parts'):
                                logger.error(f"      Parts: {content_item.parts}")
            raise ValueError(f"Billing agent returned empty response. Events: {len(billing_events)}")
        else:
            raise ValueError("Billing agent returned no events")
        billing_content = clean_json_response(billing_content)
        logger.info(f"Billing agent response length: {len(billing_content)} chars, preview: {billing_content[:200]}")
        billing_result = json.loads(billing_content)
        validate_billing_result(billing_result)
        logger.info(f"Step 2 complete: Billing agent risk_score={billing_result.get('billing_risk_score', 0)}")
        
        # Step 3: Run discharge agent with identity and billing results
        logger.info(f"Step 3: Running discharge agent for patient {patient_id}")
        
        # Create session for discharge agent with unique ID
        discharge_session_id = f"session_{patient_id}_discharge_{session_suffix}"
        discharge_session = session_service.create_session_sync(
            app_name="discharge_agent",
            user_id=f"user_{patient_id}",
            session_id=discharge_session_id
        )
        
        # Run discharge agent with identity and billing results
        discharge_prompt = f"""Assess discharge readiness for patient {patient_id}.

Identity Analysis Results: {json.dumps(identity_result)}
Billing Analysis Results: {json.dumps(billing_result)}

Use the tools to check for active encounters, pending procedures, and fraud blockers.
Determine if patient is ready for discharge and identify any blockers."""

        discharge_message = types.Content(
            parts=[types.Part(text=discharge_prompt)],
            role="user"
        )
        
        discharge_events = []
        async for event in discharge_runner.run_async(
            user_id=f"user_{patient_id}",
            session_id=discharge_session_id,
            new_message=discharge_message
        ):
            discharge_events.append(event)
            # Log event details for debugging
            event_type = event.__class__.__name__ if hasattr(event, '__class__') else str(type(event))
            logger.debug(f"Discharge event {len(discharge_events)}: {event_type}")
            if hasattr(event, 'content') and event.content:
                logger.debug(f"  Event has content: {type(event.content)}")
        
        logger.info(f"Discharge agent completed with {len(discharge_events)} events")
        
        # If last event has None content, agent may need a follow-up to generate response
        needs_followup = False
        if discharge_events:
            last_event = discharge_events[-1]
            if hasattr(last_event, 'content'):
                if last_event.content is None:
                    needs_followup = True
                elif isinstance(last_event.content, list) and len(last_event.content) == 0:
                    needs_followup = True
        
        if needs_followup:
            logger.warning("Last discharge event has None content - sending follow-up to get JSON response")
            followup_message = types.Content(
                parts=[types.Part(text="Please provide your analysis result as JSON. Return ONLY the JSON object with discharge_ready, blockers, and delay_hours fields. Do not include markdown or explanations.")],
                role="user"
            )
            followup_events = []
            async for event in discharge_runner.run_async(
                user_id=f"user_{patient_id}",
                session_id=discharge_session_id,
                new_message=followup_message
            ):
                followup_events.append(event)
                logger.debug(f"Followup discharge event {len(followup_events)}: {type(event).__name__}")
            
            # Add followup events to the main list
            discharge_events.extend(followup_events)
            logger.info(f"Added {len(followup_events)} followup events. Total events: {len(discharge_events)}")
        
        # Try to extract response from events, prioritizing model response events
        discharge_content = None
        all_text = []
        
        # First, try to find a model response event (after function calls)
        # Iterate in reverse to get the most recent response first
        for i, event in enumerate(reversed(discharge_events)):
            if hasattr(event, 'content') and event.content:
                content_list = event.content if isinstance(event.content, list) else [event.content]
                for content_item in content_list:
                    if content_item and hasattr(content_item, 'parts') and content_item.parts:
                        for part in content_item.parts:
                            # Look for text parts (skip function_call parts)
                            if hasattr(part, 'text') and part.text:
                                text = str(part.text).strip()
                                if text:
                                    logger.debug(f"Found text in discharge event {len(discharge_events) - i - 1}: {text[:100]}...")
                                    all_text.append(text)
                                    if not discharge_content:  # Use first non-empty text found
                                        discharge_content = text
        
        # If we found text, join it all
        if all_text:
            discharge_content = " ".join(all_text)
            logger.info(f"Extracted discharge content from events: {len(discharge_content)} chars")
        elif discharge_events:
            # If no text found, log detailed event information for debugging
            logger.error(f"Discharge agent returned no text content. Event count: {len(discharge_events)}")
            for i, event in enumerate(discharge_events):
                event_type = type(event).__name__ if hasattr(type(event), '__name__') else str(type(event))
                logger.error(f"Event {i}: type={event_type}, has content={hasattr(event, 'content')}")
                if hasattr(event, 'content'):
                    logger.error(f"  Content type: {type(event.content)}, value: {event.content}")
                    if event.content and isinstance(event.content, list):
                        for j, content_item in enumerate(event.content):
                            logger.error(f"    Content item {j}: {type(content_item)}")
                            if content_item and hasattr(content_item, 'parts'):
                                logger.error(f"      Parts: {content_item.parts}")
            raise ValueError(f"Discharge agent returned empty response. Events: {len(discharge_events)}")
        else:
            raise ValueError("Discharge agent returned no events")
        
        discharge_content = clean_json_response(discharge_content)
        logger.info(f"Discharge agent response length: {len(discharge_content)} chars, preview: {discharge_content[:200]}")
        discharge_result = json.loads(discharge_content)
        validate_discharge_result(discharge_result)
        logger.info(f"Step 3 complete: Discharge agent - ready={discharge_result.get('discharge_ready', False)}, blockers={len(discharge_result.get('blockers', []))}")
        
        # Get raw data for final result
        raw_data = fetch_patient_data_direct(patient_id)
        
        # Combine all results
        result = {
            "patient_id": patient_id,
            "identity": identity_result,
            "billing": billing_result,
            "discharge": discharge_result,
            "raw": raw_data,
            "final": {
                **identity_result,
                **billing_result,
                **discharge_result
            }
        }
        
        duration = (time.time() - start_time) * 1000
        logger.info(f"Workflow analysis complete: duration={duration:.2f}ms, fraud_score={identity_result.get('fraud_risk_score', 0)}")
        
        return result
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {str(e)}")
        raise ValueError(f"Failed to parse agent response as JSON: {str(e)}")
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise
    except Exception as e:
        duration = (time.time() - start_time) * 1000
        logger.error(f"Unexpected error during workflow analysis: {str(e)}, duration={duration:.2f}ms", exc_info=True)
        raise Exception(f"Error analyzing patient: {str(e)}")

def analyze_agent1_only(patient_id: str) -> Dict[str, Any]:
    """
    Analyze a patient using only Agent 1 (Identity & Claims Fraud Detection) with ADK.
    
    Args:
        patient_id: Patient UUID to analyze
        
    Returns:
        Dictionary with analysis results containing:
        - fraud_risk_score: Number 0-100
        - identity_misuse_flag: Boolean
        - reasons: List of strings
        
    Raises:
        ValueError: If patient not found or analysis fails
        Exception: For unexpected errors
    """
    try:
        return asyncio.run(analyze_agent1_only_async(patient_id))
    finally:
        # Clean up any pending async tasks
        try:
            loop = asyncio.get_event_loop()
            if not loop.is_closed():
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
        except:
            pass

async def analyze_agent1_only_async(patient_id: str) -> Dict[str, Any]:
    """
    Async implementation of analyze_agent1_only.
    """
    start_time = time.time()
    logger.info(f"Starting Agent 1 analysis for patient: {patient_id}")
    
    try:
        # Create prompt that instructs agent to use the tool
        prompt = f"""Analyze patient {patient_id} for fraud and identity misuse.

First, use the fetch_patient_data_tool to retrieve the patient's data.
Then analyze the data for fraud patterns and return the results in the required JSON format."""

        # Create session first
        session = session_service.create_session_sync(
            app_name="identity_agent",
            user_id=f"user_{patient_id}",
            session_id=f"session_{patient_id}"
        )
        
        # Run the identity agent using Runner
        # Create Content with the prompt
        message_content = types.Content(
            parts=[types.Part(text=prompt)],
            role="user"
        )
        response_events = []
        async for event in identity_runner.run_async(
            user_id=f"user_{patient_id}",
            session_id=f"session_{patient_id}",
            new_message=message_content
        ):
            response_events.append(event)
        
        # Extract response from events - find final response event
        final_response_event = None
        for event in response_events:
            if hasattr(event, 'is_final_response') and event.is_final_response():
                final_response_event = event
                break
        if final_response_event is None and response_events:
            final_response_event = response_events[-1]
        
        # Debug: log event structure
        if final_response_event:
            logger.debug(f"Final event type: {type(final_response_event)}")
            logger.debug(f"Final event has content: {hasattr(final_response_event, 'content')}")
            if hasattr(final_response_event, 'content'):
                logger.debug(f"Content type: {type(final_response_event.content)}, length: {len(final_response_event.content) if isinstance(final_response_event.content, list) else 'N/A'}")
        
        # Extract the response content - try multiple approaches
        content = parse_agent_response(final_response_event)
        
        # If content is empty, try extracting from all events
        if not content or not content.strip():
            logger.info("Content empty from final event, trying all events...")
            all_text = []
            for i, event in enumerate(response_events):
                if hasattr(event, 'content') and event.content:
                    content_list = event.content if isinstance(event.content, list) else [event.content]
                    for content_item in content_list:
                        if hasattr(content_item, 'parts') and content_item.parts:
                            for part in content_item.parts:
                                # Check for text attribute - this is the key
                                if hasattr(part, 'text') and part.text and part.text.strip():
                                    all_text.append(part.text.strip())
                                    logger.debug(f"Event {i}: Found text part ({len(part.text)} chars)")
            if all_text:
                # Join all text parts
                content = " ".join(all_text)
                logger.info(f"Extracted content from all events: {len(content)} chars")
            else:
                logger.warning("No text found in any event parts")
        
        if not content or not content.strip():
            # Log the event structure for debugging
            logger.error(f"No content extracted. Event structure: {[type(e).__name__ for e in response_events]}")
            raise ValueError("No content extracted from agent response. Agent may not have returned text.")
        
        content = clean_json_response(content)
        
        # Parse JSON
        try:
            result = json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error. Content was: {content[:200]}")
            raise
        
        # Validate output structure
        validate_identity_result(result)
        
        duration = (time.time() - start_time) * 1000
        logger.info(f"Agent 1 analysis complete: duration={duration:.2f}ms, fraud_score={result['fraud_risk_score']}, identity_misuse={result['identity_misuse_flag']}")
        
        return result
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {str(e)}")
        raise ValueError(f"Failed to parse agent response as JSON: {str(e)}")
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise
    except Exception as e:
        duration = (time.time() - start_time) * 1000
        logger.error(f"Unexpected error during Agent 1 analysis: {str(e)}, duration={duration:.2f}ms", exc_info=True)
        raise Exception(f"Error analyzing patient: {str(e)}")

def fetch_patient_data(patient_id: str) -> Dict[str, Any]:
    """
    Fetch patient data (backward compatibility wrapper for API server).
    
    Args:
        patient_id: Patient UUID
        
    Returns:
        Dictionary with patient, claims, and claim_lines
    """
    return fetch_patient_data_direct(patient_id)

def get_sample_patient_ids(limit: int = 10) -> list:
    """Get sample patient IDs for testing"""
    return patients.index.tolist()[:limit]

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("MediGuard AI Agent (ADK) - Starting Analysis...")
    
    # Get patient ID from command line or prompt
    if len(sys.argv) > 1:
        patient_id = sys.argv[1]
    else:
        # Show sample patient IDs
        sample_ids = get_sample_patient_ids(5)
        print(f"\nSample Patient IDs (first 5):")
        for i, pid in enumerate(sample_ids, 1):
            print(f"  {i}. {pid}")
        
        patient_id = input("\nEnter Patient ID (UUID): ").strip()
        if not patient_id:
            print("ERROR: No patient ID provided. Exiting.")
            sys.exit(1)
    
        print(f"\nAnalyzing Patient: {patient_id}\n")
    
    try:
        # Run Agent 1 only
        result = analyze_agent1_only(patient_id)
        
        print("=" * 60)
        print("AGENT 1 ANALYSIS RESULTS (Claims & Identity Fraud)")
        print("=" * 60)
        print(json.dumps(result, indent=2))
    except ValueError as e:
        print(f"ERROR: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Unexpected error: {str(e)}")
        sys.exit(1)
