"""
FastAPI server for MediGuard AI Frontend
Provides REST API endpoints for the Next.js frontend
Uses Google ADK instead of LangChain/LangGraph
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import json
import sys
import os
import time
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import asyncio

# Add parent directory to path to import main
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import (
    analyze_patient,
    analyze_agent1_only,
    get_sample_patient_ids,
    fetch_patient_data
)

# Setup logging
logger = logging.getLogger(__name__)

app = FastAPI(title="MediGuard AI API (ADK)", version="2.0.0")

# CORS middleware for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Log request
    logger.info(json.dumps({
        "event": "api_request",
        "method": request.method,
        "path": request.url.path,
        "timestamp": datetime.utcnow().isoformat()
    }))
    
    response = await call_next(request)
    
    # Log response
    duration = (time.time() - start_time) * 1000
    logger.info(json.dumps({
        "event": "api_response",
        "method": request.method,
        "path": request.url.path,
        "status_code": response.status_code,
        "duration_ms": duration,
        "timestamp": datetime.utcnow().isoformat()
    }))
    
    return response

class PatientAnalysisRequest(BaseModel):
    patient_id: str

class AnalysisResponse(BaseModel):
    identity: Optional[dict] = None
    billing: Optional[dict] = None
    discharge: Optional[dict] = None
    final: Optional[dict] = None

@app.get("/")
async def root():
    return {"message": "MediGuard AI API (ADK)", "version": "2.0.0", "framework": "Google ADK"}

@app.get("/api/sample-ids")
async def get_sample_ids(limit: int = 10):
    """Get sample patient IDs for testing"""
    try:
        ids = get_sample_patient_ids(limit)
        return {"ids": ids}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_patient_endpoint(request: PatientAnalysisRequest):
    """Analyze a patient through all agents"""
    try:
        logger.info(f"Received analysis request for patient: {request.patient_id}")
        
        # Validate patient exists
        try:
            fetch_patient_data(request.patient_id)
            logger.info(f"Patient {request.patient_id} validated successfully")
        except ValueError as e:
            logger.error(f"Patient validation failed: {str(e)}")
            raise HTTPException(status_code=404, detail=str(e))
        
        # Run full analysis - get the full workflow state, not just final
        # Use thread pool executor to run sync function that uses asyncio.run()
        logger.info(f"Starting analysis for patient: {request.patient_id}")
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            workflow_result = await loop.run_in_executor(executor, analyze_patient, request.patient_id)
        logger.info(f"Analysis completed for patient: {request.patient_id}")
        
        # Extract results from workflow state
        identity_result = workflow_result.get("identity", {})
        billing_result = workflow_result.get("billing", {})
        discharge_result = workflow_result.get("discharge", {})
        final_result = workflow_result.get("final", {})
        
        logger.info(f"Extracted results - identity: {bool(identity_result)}, billing: {bool(billing_result)}, discharge: {bool(discharge_result)}")
        
        # Structure response to match frontend expectations
        response = AnalysisResponse(
            identity={
                "fraud_risk_score": identity_result.get("fraud_risk_score", 0),
                "identity_misuse_flag": identity_result.get("identity_misuse_flag", False),
                "reasons": identity_result.get("reasons", [])
            } if identity_result else None,
            billing={
                "billing_fraud_score": billing_result.get("billing_fraud_score", billing_result.get("billing_risk_score", 0)),
                "billing_flags": billing_result.get("billing_flags", billing_result.get("suspicious_items", [])),
                "billing_explanation": billing_result.get("billing_explanation", billing_result.get("explanations", ""))
            } if billing_result else None,
            discharge={
                "discharge_ready": discharge_result.get("discharge_ready", discharge_result.get("discharge_ready_flag", False)),
                "blockers": discharge_result.get("blockers", []),
                "delay_hours": discharge_result.get("delay_hours", 0),
                "priority_level": discharge_result.get("priority_level", "medium")
            } if discharge_result else None,
            final=final_result if final_result else {}
        )
        
        logger.info(f"Response structured successfully for patient: {request.patient_id}")
        return response
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"Analysis error: {str(e)}\n{traceback.format_exc()}"
        logger.error(f"Error in analyze_patient_endpoint: {error_detail}")
        raise HTTPException(status_code=500, detail=error_detail)

@app.post("/api/analyze/agent1")
async def analyze_agent1_only_endpoint(request: PatientAnalysisRequest):
    """Analyze a patient using only Agent 1"""
    try:
        # Use thread pool executor to run sync function that uses asyncio.run()
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            result = await loop.run_in_executor(executor, analyze_agent1_only, request.patient_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        import traceback
        error_detail = f"Analysis error: {str(e)}\n{traceback.format_exc()}"
        logger.error(f"Error in analyze_agent1_only_endpoint: {error_detail}")
        raise HTTPException(status_code=500, detail=error_detail)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "framework": "Google ADK"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

