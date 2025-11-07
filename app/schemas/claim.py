from datetime import date
from typing import List, Optional, Literal
from pydantic import BaseModel, Field, validator
from enum import Enum

class DocumentType(str, Enum):
    BILL = "bill"
    DISCHARGE_SUMMARY = "discharge_summary"
    ID_CARD = "id_card"
    PRESCRIPTION = "prescription"
    LAB_REPORT = "lab_report"
    UNKNOWN = "unknown"

class BillDocument(BaseModel):
    type: Literal[DocumentType.BILL] = DocumentType.BILL
    hospital_name: Optional[str] = Field(..., description="Name of the hospital or medical facility")
    total_amount: float = Field(..., gt=0, description="Total amount billed")
    date_of_service: date = Field(..., description="Date of service in YYYY-MM-DD format")
    patient_name: Optional[str] = Field(None, description="Name of the patient")
    patient_id: Optional[str] = Field(None, description="Patient ID or MRN")
    items: Optional[List[dict]] = Field(default_factory=list, description="List of billed items")
    diagnosis_codes: Optional[List[str]] = Field(default_factory=list, description="List of diagnosis codes")
    procedure_codes: Optional[List[str]] = Field(default_factory=list, description="List of procedure codes")

class DischargeSummaryDocument(BaseModel):
    type: Literal[DocumentType.DISCHARGE_SUMMARY] = DocumentType.DISCHARGE_SUMMARY
    patient_name: str = Field(..., description="Name of the patient")
    patient_id: Optional[str] = Field(None, description="Patient ID or MRN")
    admission_date: date = Field(..., description="Date of admission in YYYY-MM-DD format")
    discharge_date: date = Field(..., description="Date of discharge in YYYY-MM-DD format")
    diagnosis: str = Field(..., description="Primary diagnosis")
    secondary_diagnoses: Optional[List[str]] = Field(default_factory=list, description="Secondary diagnoses")
    procedures: Optional[List[str]] = Field(default_factory=list, description="Procedures performed")
    discharge_instructions: Optional[str] = Field(None, description="Discharge instructions")

class IdCardDocument(BaseModel):
    type: Literal[DocumentType.ID_CARD] = DocumentType.ID_CARD
    insurance_provider: str = Field(..., description="Name of the insurance provider")
    policy_number: str = Field(..., description="Insurance policy number")
    group_number: Optional[str] = Field(None, description="Group number if applicable")
    member_id: str = Field(..., description="Member/Subscriber ID")
    member_name: str = Field(..., description="Name of the insured member")
    relationship: Optional[str] = Field("self", description="Relationship to the primary policyholder")
    effective_date: Optional[date] = Field(None, description="Effective date of coverage")
    expiration_date: Optional[date] = Field(None, description="Expiration date of coverage")

class PrescriptionDocument(BaseModel):
    type: Literal[DocumentType.PRESCRIPTION] = DocumentType.PRESCRIPTION
    patient_name: str = Field(..., description="Name of the patient")
    date_prescribed: date = Field(..., description="Date the prescription was written")
    medications: List[dict] = Field(..., description="List of prescribed medications")
    prescriber_name: Optional[str] = Field(None, description="Name of the prescribing doctor")
    prescriber_license: Optional[str] = Field(None, description="License number of the prescriber")
    instructions: Optional[str] = Field(None, description="Usage instructions")

class LabReportDocument(BaseModel):
    type: Literal[DocumentType.LAB_REPORT] = DocumentType.LAB_REPORT
    patient_name: str = Field(..., description="Name of the patient")
    patient_id: Optional[str] = Field(None, description="Patient ID or MRN")
    date_collected: date = Field(..., description="Date the sample was collected")
    date_reported: date = Field(..., description="Date the results were reported")
    test_results: List[dict] = Field(..., description="List of test results")
    lab_name: Optional[str] = Field(None, description="Name of the laboratory")
    ordering_physician: Optional[str] = Field(None, description="Name of the ordering physician")

from pydantic import RootModel

class ClaimDocument(RootModel):
    """Union of all possible document types."""
    root: BillDocument | DischargeSummaryDocument | IdCardDocument | PrescriptionDocument | LabReportDocument

class ClaimValidation(BaseModel):
    """Validation results for the claim."""
    missing_documents: List[str] = Field(default_factory=list, description="List of missing required document types")
    discrepancies: List[dict] = Field(default_factory=list, description="List of data discrepancies found")
    is_valid: bool = Field(..., description="Whether the claim is valid")

class ClaimDecision(BaseModel):
    """Final claim decision."""
    status: Literal["approved", "rejected", "pending"]
    reason: str = Field(..., description="Explanation for the decision")
    amount_approved: Optional[float] = Field(None, description="Approved amount if applicable")
    amount_rejected: Optional[float] = Field(None, description="Rejected amount if applicable")

class ProcessedClaim(BaseModel):
    """Complete claim processing result."""
    documents: List[ClaimDocument] = Field(..., description="List of processed documents")
    validation: ClaimValidation = Field(..., description="Validation results")
    decision: ClaimDecision = Field(..., description="Claim decision")
