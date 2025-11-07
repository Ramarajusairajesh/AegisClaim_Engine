from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
import asyncio
from datetime import datetime
from loguru import logger

from app.schemas.claim import (
    DocumentType,
    ClaimDocument,
    ClaimValidation,
    ClaimDecision,
    ProcessedClaim
)
from .agents.classifier_agent import ClassifierAgent
from .agents.bill_agent import BillAgent
from .agents.discharge_agent import DischargeAgent
from .agents.id_card_agent import IdCardAgent
from .agents.prescription_agent import PrescriptionAgent
from .agents.lab_report_agent import LabReportAgent

class ClaimProcessor:
    """
    Processes insurance claims by coordinating multiple specialized agents.
    Handles document classification, data extraction, validation, and claim decisions.
    """
    
    def __init__(self):
        # Initialize all agents
        self.classifier = ClassifierAgent()
        self.agents = {
            DocumentType.BILL: BillAgent(),
            DocumentType.DISCHARGE_SUMMARY: DischargeAgent(),
            DocumentType.ID_CARD: IdCardAgent(),
            DocumentType.PRESCRIPTION: PrescriptionAgent(),
            DocumentType.LAB_REPORT: LabReportAgent(),
        }
        
        # Required document types for claim processing
        self.required_docs = {
            DocumentType.BILL,
            DocumentType.ID_CARD
        }
        
        # Document type descriptions for validation messages
        self.doc_descriptions = {
            DocumentType.BILL: "medical bill",
            DocumentType.DISCHARGE_SUMMARY: "discharge summary",
            DocumentType.ID_CARD: "insurance ID card",
            DocumentType.PRESCRIPTION: "prescription",
            DocumentType.LAB_REPORT: "lab report"
        }
    
    async def process_claim(self, file_paths: List[Path]) -> ProcessedClaim:
        """
        Process a claim with multiple documents.
        
        Args:
            file_paths: List of paths to document files
            
        Returns:
            ProcessedClaim object with extracted data and decision
        """
        start_time = datetime.now()
        logger.info(f"Starting claim processing for {len(file_paths)} documents")
        
        # Step 1: Classify all documents
        classified_docs = await self._classify_documents(file_paths)
        
        # Step 2: Process each document with the appropriate agent
        processed_docs = await self._process_documents(classified_docs)
        
        # Step 3: Validate the claim
        validation = self._validate_claim(processed_docs)
        
        # Step 4: Make a claim decision
        decision = self._make_decision(processed_docs, validation)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Claim processing completed in {processing_time:.2f} seconds")
        logger.info(f"Decision: {decision.status} - {decision.reason}")
        
        return ProcessedClaim(
            documents=processed_docs,
            validation=validation,
            decision=decision,
            metadata={
                "processing_time_seconds": processing_time,
                "processed_at": datetime.now().isoformat()
            }
        )
    
    async def _classify_documents(self, file_paths: List[Path]) -> List[Tuple[Path, DocumentType]]:
        """Classify each document using the ClassifierAgent."""
        tasks = []
        for file_path in file_paths:
            tasks.append(self._classify_single_doc(file_path))
        return await asyncio.gather(*tasks)
    
    async def _classify_single_doc(self, file_path: Path) -> Tuple[Path, DocumentType]:
        """Classify a single document."""
        try:
            doc_type = await self.classifier.classify_document(file_path)
            logger.info(f"Classified {file_path.name} as {doc_type.name}")
            return (file_path, doc_type)
        except Exception as e:
            logger.error(f"Error classifying {file_path}: {str(e)}")
            return (file_path, DocumentType.UNKNOWN)
    
    async def _process_documents(self, classified_docs: List[Tuple[Path, DocumentType]]) -> List[ClaimDocument]:
        """Process each document with the appropriate agent."""
        processed = []
        
        for file_path, doc_type in classified_docs:
            if doc_type == DocumentType.UNKNOWN:
                logger.warning(f"Skipping unknown document type: {file_path}")
                continue
                
            try:
                agent = self.agents.get(doc_type)
                if not agent:
                    logger.warning(f"No agent available for document type: {doc_type}")
                    continue
                
                # Process the document with the appropriate agent
                result = await agent.process(file_path)
                
                # Add file metadata
                result_dict = result.dict()
                result_dict['file_name'] = file_path.name
                result_dict['file_size'] = file_path.stat().st_size
                
                processed.append(ClaimDocument.parse_obj(result_dict))
                logger.info(f"Processed {file_path.name} as {doc_type.name}")
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                # Create a minimal document with error information
                processed.append(ClaimDocument.parse_obj({
                    "type": doc_type,
                    "file_name": file_path.name,
                    "error": str(e),
                    "status": "error"
                }))
        
        return processed
    
    def _validate_claim(self, documents: List[ClaimDocument]) -> ClaimValidation:
        """Validate the claim based on the processed documents."""
        missing_docs = []
        discrepancies = []
        
        # Check for missing required documents
        found_types = {doc.type for doc in documents if hasattr(doc, 'type')}
        for doc_type in self.required_docs:
            if doc_type not in found_types:
                missing_docs.append(self.doc_descriptions.get(doc_type, doc_type.name))
        
        # Check for data consistency across documents
        if len(documents) > 1:
            # Example: Verify patient names match across documents
            patient_names = {}
            for doc in documents:
                if hasattr(doc, 'patient_name') and doc.patient_name:
                    doc_type = self.doc_descriptions.get(doc.type, doc.type)
                    patient_names[doc_type] = doc.patient_name
            
            # If we have multiple patient names, check for mismatches
            if len(set(patient_names.values())) > 1:
                discrepancy = {
                    "field": "patient_name",
                    "message": "Patient name mismatch across documents",
                    "details": patient_names
                }
                discrepancies.append(discrepancy)
        
        return ClaimValidation(
            missing_documents=missing_docs,
            discrepancies=discrepancies,
            is_valid=len(missing_docs) == 0 and len(discrepancies) == 0
        )
    
    def _make_decision(self, 
                      documents: List[ClaimDocument], 
                      validation: ClaimValidation) -> ClaimDecision:
        """Make a claim decision based on the processed documents and validation."""
        # If there are missing required documents, reject the claim
        if validation.missing_documents:
            missing_list = ", ".join(validation.missing_documents)
            return ClaimDecision(
                status="rejected",
                reason=f"Missing required documents: {missing_list}",
                amount_approved=0.0,
                amount_rejected=self._calculate_total_amount(documents)
            )
        
        # If there are data discrepancies, flag for review
        if validation.discrepancies:
            return ClaimDecision(
                status="pending",
                reason="Data discrepancies found, requires manual review",
                amount_approved=0.0,
                amount_rejected=0.0
            )
        
        # Simple approval logic - in a real system, this would be more sophisticated
        try:
            total_amount = self._calculate_total_amount(documents)
            
            # Example: Approve claims under $10,000 automatically
            if total_amount <= 10000.0:
                return ClaimDecision(
                    status="approved",
                    reason="Claim meets all requirements",
                    amount_approved=total_amount,
                    amount_rejected=0.0
                )
            else:
                return ClaimDecision(
                    status="pending",
                    reason="Claim amount exceeds automatic approval limit",
                    amount_approved=0.0,
                    amount_rejected=0.0
                )
                
        except Exception as e:
            logger.error(f"Error making claim decision: {str(e)}")
            return ClaimDecision(
                status="rejected",
                reason=f"Error processing claim: {str(e)}",
                amount_approved=0.0,
                amount_rejected=self._calculate_total_amount(documents)
            )
    
    def _calculate_total_amount(self, documents: List[ClaimDocument]) -> float:
        """Calculate the total claim amount from all documents."""
        total = 0.0
        
        for doc in documents:
            if hasattr(doc, 'total_amount') and doc.total_amount is not None:
                try:
                    total += float(doc.total_amount)
                except (ValueError, TypeError):
                    continue
                    
        return total
