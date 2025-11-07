from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from pathlib import Path
import tempfile
import uuid
import os
from typing import List
import shutil

from app.services.claim_processor import ClaimProcessor
from app.schemas.claim import ProcessedClaim
from app.core.config import settings

router = APIRouter()

# Ensure upload directory exists
UPLOAD_DIR = Path(settings.UPLOAD_DIR)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@router.post("/process-claim", response_model=ProcessedClaim)
async def process_claim(
    files: List[UploadFile] = File(..., description="List of claim documents (PDF, JPG, PNG, etc.)")
):
    """
    Process an insurance claim with multiple documents.
    
    This endpoint accepts multiple document files, classifies them, extracts relevant information,
    and returns a claim decision with extracted data.
    
    - **files**: List of document files (bills, ID cards, prescriptions, etc.)
    
    Returns a structured response with processed documents, validation results, and claim decision.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    # Create a temporary directory for this request
    temp_dir = UPLOAD_DIR / str(uuid.uuid4())
    temp_dir.mkdir()
    
    try:
        file_paths = []
        
        # Save uploaded files to temporary directory
        for file in files:
            # Generate a safe filename
            file_ext = Path(file.filename).suffix if file.filename else ".bin"
            safe_filename = f"{uuid.uuid4()}{file_ext}"
            file_path = temp_dir / safe_filename
            
            # Save the file
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            file_paths.append(file_path)
        
        # Process the claim
        processor = ClaimProcessor()
        result = await processor.process_claim(file_paths)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing claim: {str(e)}")
        
    finally:
        # Clean up temporary files
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as e:
            print(f"Warning: Error cleaning up temporary files: {str(e)}")
            
        # Close all file handles
        for file in files:
            if hasattr(file.file, 'close'):
                file.file.close()

@router.get("/supported-document-types")
async def get_supported_document_types():
    """
    Get a list of supported document types and their descriptions.
    """
    return {
        "bill": "Medical bill or invoice",
        "discharge_summary": "Hospital discharge summary",
        "id_card": "Insurance ID card",
        "prescription": "Medical prescription",
        "lab_report": "Laboratory test results"
    }
