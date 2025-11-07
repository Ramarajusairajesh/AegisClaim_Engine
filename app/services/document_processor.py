from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypeAlias

from fastapi import UploadFile
from loguru import logger

from ..schemas.document import (
    BillDocument,
    ClaimDecision,
    ClaimResponse,
    DischargeSummaryDocument,
    DocumentType,
    IDCardDocument,
    ValidationResult,
)

# Type aliases
UploadedFiles: TypeAlias = Sequence[UploadFile]
ProcessedDocument: TypeAlias = BillDocument | DischargeSummaryDocument | IDCardDocument
from ..config import settings
from .agents.classifier_agent import ClassifierAgent
from .agents.bill_agent import BillAgent
from .agents.discharge_agent import DischargeAgent

class DocumentProcessor:
    """
    Handles the end-to-end processing of claim documents.
    
    Attributes:
        classifier_agent: Agent for document classification
        bill_agent: Agent for processing medical bills
        discharge_agent: Agent for processing discharge summaries
        upload_dir: Directory to store uploaded files
    """
    
    def __init__(
        self,
        classifier_agent: ClassifierAgent,
        bill_agent: BillAgent,
        discharge_agent: DischargeAgent,
    ) -> None:
        """Initialize the DocumentProcessor with the required agents."""
        self.classifier_agent = classifier_agent
        self.bill_agent = bill_agent
        self.discharge_agent = discharge_agent
        self.upload_dir = Path(settings.UPLOAD_FOLDER)
        self.upload_dir.mkdir(exist_ok=True, parents=True)
    
    async def process_documents(self, files: UploadedFiles) -> ClaimResponse:
        """
        Process a list of uploaded documents and return a claim decision.

        Args:
            files: Sequence of uploaded files to process

        Returns:
            ClaimResponse: The processed claim information and decision

        Raises:
            HTTPException: If there's an error processing the documents
        """
        try:
            # Save and validate files
            saved_files = await self._save_uploaded_files(files)
            
            # Process each document concurrently
            tasks = [self._process_single_file(file_path) for file_path in saved_files]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out any processing errors
            processed_docs: list[ProcessedDocument] = [
                doc for doc in results 
                if not isinstance(doc, (Exception, type(None)))
            ]
            
            # Log any processing errors
            errors = [
                str(e) for e in results 
                if isinstance(e, Exception)
            ]
            
            if errors:
                logger.warning(f"Encountered {len(errors)} errors during processing")
            
            # Validate the claim
            validation = self._validate_claim(processed_docs)
            
            # Make claim decision
            decision = self._make_claim_decision(processed_docs, validation)
            
            # Prepare response
            return ClaimResponse(
                documents=[doc.model_dump() for doc in processed_docs if doc],
                validation=validation.model_dump(),
                claim_decision=decision,
                metadata={
                    "documents_processed": len(processed_docs),
                    "documents_received": len(files),
                    "processing_errors": errors if errors else None
                }
            )
            
        except Exception as e:
            logger.opt(exception=e).error("Error processing documents")
            raise
    
    async def _save_uploaded_files(self, files: UploadedFiles) -> list[Path]:
        """
        Save uploaded files to disk and return their paths.

        Args:
            files: Sequence of uploaded files to save

        Returns:
            List of paths where files were saved
        """
        saved_paths: list[Path] = []
        
        for file in files:
            if not file.filename:
                logger.warning("Skipping file with no filename")
                continue

            try:
                # Validate file extension
                file_ext = Path(file.filename).suffix.lower()
                if file_ext not in [f".{ext}" for ext in settings.ALLOWED_EXTENSIONS]:
                    logger.warning(f"Skipping file with invalid extension: {file.filename}")
                    continue
                
                # Create unique filename
                file_id = uuid.uuid4()
                file_path = self.upload_dir / f"{file_id}{file_ext}"
                
                # Save file
                try:
                    content = await file.read()
                    file_path.write_bytes(content)
                    logger.info("Saved uploaded file: {}", file_path)
                    saved_paths.append(file_path)
                except OSError as e:
                    logger.error("Error saving file {}: {}", file.filename, str(e))
                    continue
                
            except Exception as e:
                logger.opt(exception=e).error("Unexpected error processing file {}", file.filename)
                continue
                
        if not saved_paths:
            raise ValueError("No valid files were uploaded")
            
        return saved_paths
    
    async def _process_single_file(self, file_path: Path) -> ProcessedDocument | None:
        """
        Process a single document file.

        Args:
            file_path: Path to the document file to process

        Returns:
            Processed document or None if processing failed

        Raises:
            AgentError: If there's an error during processing
        """
        try:
            # Classify document
            doc_type = await self.classifier_agent.classify_document(file_path)
            
            # Process based on document type
            match doc_type:
                case DocumentType.BILL:
                    return await self.bill_agent.process(file_path)
                case DocumentType.DISCHARGE_SUMMARY:
                    return await self.discharge_agent.process(file_path)
                case DocumentType.ID_CARD:
                    # TODO: Implement ID card processing
                    logger.warning("ID card processing not yet implemented")
                    return None
                case _:
                    logger.warning("No processor available for document type: {}", doc_type)
                    return None
                
        except Exception as e:
            logger.opt(exception=e).error("Error processing file {}", file_path)
            raise AgentError(f"Failed to process {file_path.name}") from e
    
    def _validate_claim(self, documents: Sequence[ProcessedDocument]) -> ValidationResult:
        """
        Validate the claim based on processed documents.

        Args:
            documents: List of processed documents to validate

        Returns:
            Validation result with any errors found
        """
        errors: list[str] = []
        
        # Check for required documents
        doc_types = {doc.type for doc in documents}
        missing_docs = [
            doc_type.value 
            for doc_type in [
                DocumentType.BILL, 
                DocumentType.DISCHARGE_SUMMARY, 
                DocumentType.ID_CARD
            ] 
            if doc_type not in doc_types
        ]
        
        if missing_docs:
            errors.append(f"Missing required documents: {', '.join(missing_docs)}")
        
        # Cross-validate data between documents
        self._cross_validate_documents(documents, errors)
        
        return ValidationResult(
            is_valid=not errors,
            errors=errors or None  # Use None instead of empty list for cleaner JSON
        )
    
    def _cross_validate_documents(
        self, 
        documents: Sequence[ProcessedDocument],
        errors: list[str]
    ) -> None:
        """
        Cross-validate data between different documents.
        
        Args:
            documents: List of processed documents
            errors: List to append any validation errors to
        """
        # Extract relevant data for cross-validation
        patient_names = {
            doc.patient_name 
            for doc in documents 
            if hasattr(doc, 'patient_name') and doc.patient_name
        }
        
        # Check for consistent patient names across documents
        if len(patient_names) > 1:
            errors.append(
                f"Inconsistent patient names found: {', '.join(patient_names)}"
            )
    
    def _make_claim_decision(
        self, 
        documents: Sequence[ProcessedDocument], 
        validation: ValidationResult
    ) -> ClaimDecision:
        """
        Make a claim decision based on processed documents and validation results.

        Args:
            documents: List of processed documents
            validation: Validation results

        Returns:
            Claim decision with status and reasoning
        """
        if not validation.is_valid:
            return ClaimDecision(
                status="rejected",
                reason=f"Invalid claim: {'; '.join(validation.errors or [])}",
                confidence=0.9
            )
        
        # Basic decision logic - can be enhanced with more sophisticated rules
        return self._evaluate_claim_approval(documents)
    
    def _evaluate_claim_approval(self, documents: Sequence[ProcessedDocument]) -> ClaimDecision:
        """
        Evaluate whether to approve or reject the claim.
        
        This is a placeholder for more sophisticated claim evaluation logic.
        """
        # Placeholder for more complex decision logic
        # For now, just approve if we have the required documents
        return ClaimDecision(
            status="approved",
            reason="All required documents present and validated",
            confidence=0.95
        )
