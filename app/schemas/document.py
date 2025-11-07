from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from datetime import date

class DocumentType(str, Enum):
    """Supported document types for the claim processing system."""
    BILL = "bill"
    DISCHARGE_SUMMARY = "discharge_summary"
    ID_CARD = "id_card"
    PRESCRIPTION = "prescription"
    LAB_REPORT = "lab_report"
    UNKNOWN = "unknown"

class DocumentBase(BaseModel):
    """Base document schema."""
    type: DocumentType = Field(..., description="Type of the document")
    content: str = Field(..., description="Extracted text content from the document")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the document"
    )

class BillDocument(DocumentBase):
    """Schema for bill documents."""
    type: DocumentType = DocumentType.BILL
    hospital_name: Optional[str] = Field(None, description="Name of the hospital")
    total_amount: float = Field(..., description="Total amount in the bill")
    date_of_service: date = Field(..., description="Date of service")
    items: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of billed items with details"
    )

class DischargeSummaryDocument(DocumentBase):
    """Schema for discharge summary documents."""
    type: DocumentType = DocumentType.DISCHARGE_SUMMARY
    patient_name: str = Field(..., description="Name of the patient")
    diagnosis: str = Field(..., description="Diagnosis information")
    admission_date: date = Field(..., description="Date of admission")
    discharge_date: date = Field(..., description="Date of discharge")
    procedures: List[str] = Field(
        default_factory=list,
        description="List of medical procedures performed"
    )
    medications: List[str] = Field(
        default_factory=list,
        description="List of prescribed medications"
    )

class IDCardDocument(DocumentBase):
    """Schema for ID card documents."""
    type: DocumentType = DocumentType.ID_CARD
    patient_name: str = Field(..., description="Name of the patient")
    policy_number: str = Field(..., description="Insurance policy number")
    date_of_birth: date = Field(..., description="Patient's date of birth")
    valid_until: Optional[date] = Field(None, description="Expiration date of the ID card")

class ValidationResult(BaseModel):
    """Schema for validation results."""
    is_valid: bool = Field(..., description="Whether the document is valid")
    errors: List[str] = Field(
        default_factory=list,
        description="List of validation errors if any"
    )

class ClaimDecision(BaseModel):
    """Schema for claim decision."""
    status: str = Field(..., description="Claim status (approved/rejected)")
    reason: str = Field(..., description="Reason for the decision")
    confidence: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score of the decision (0.0 to 1.0)"
    )

class ClaimResponse(BaseModel):
    """Response schema for the /process-claim endpoint."""
    documents: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of processed documents with extracted information"
    )
    validation: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Validation results including missing documents and discrepancies"
    )
    claim_decision: ClaimDecision = Field(
        ...,
        description="Final claim decision with status and reasoning"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the claim processing"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "documents": [
                    {
                        "type": "bill",
                        "hospital_name": "ABC Hospital",
                        "total_amount": 12500,
                        "date_of_service": "2024-04-10"
                    },
                    {
                        "type": "discharge_summary",
                        "patient_name": "John Doe",
                        "diagnosis": "Fracture",
                        "admission_date": "2024-04-01",
                        "discharge_date": "2024-04-10"
                    }
                ],
                "validation": {
                    "missing_documents": [],
                    "discrepancies": []
                },
                "claim_decision": {
                    "status": "approved",
                    "reason": "All required documents present and data is consistent",
                    "confidence": 0.95
                },
                "metadata": {
                    "processing_time_ms": 2450,
                    "documents_processed": 2
                }
            }
        }
