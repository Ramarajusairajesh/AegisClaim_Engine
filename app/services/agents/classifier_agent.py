from pathlib import Path
from typing import Dict, Any, Optional, Union, List, TypedDict
import json
import re
from datetime import datetime
from loguru import logger

from .base_agent import BaseAgent, AgentError
from app.schemas.document import DocumentType, DocumentBase

# Define the structure of the extracted data
class ExtractedData(TypedDict, total=False):
    """Structure for extracted document data."""
    document_type: str
    patient_name: str
    patient_id: str
    date_of_service: str
    total_amount: float
    insurance_provider: str
    policy_number: str
    hospital_name: str
    diagnosis_codes: List[str]
    procedure_codes: List[str]
    items: List[Dict[str, Any]]
    raw_text: str

class ClassifierAgent(BaseAgent):
    """Agent responsible for classifying uploaded documents."""
    
    async def classify_document(self, file_path: Path) -> DocumentType:
        """
        Classify a document based on its content and filename.
        
        Args:
            file_path: Path to the document to classify
            
        Returns:
            DocumentType: The classified document type
        """
        try:
            # Extract text from the document
            text = await self._extract_text(file_path)
            
            # Prepare the classification prompt
            prompt = self._create_classification_prompt(file_path.name, text[:2000])  # Use first 2000 chars for classification
            
            # Get classification from LLM
            response = await self._call_llm(prompt)
            
            # Parse the response
            doc_type = self._parse_classification_response(response)
            
            logger.info(f"Classified {file_path.name} as {doc_type}")
            return doc_type
            
        except Exception as e:
            logger.error(f"Error classifying document {file_path}: {str(e)}")
            return DocumentType.UNKNOWN
    
    def _create_classification_prompt(self, filename: str, text: str) -> str:
        """Create a prompt for document classification."""
        return f"""You are an expert document classifier for medical insurance claims. 
        Your task is to classify the following document based on its filename and content.
        
        Filename: {filename}
        Document content (first 2000 chars):
        {text}
        
        Possible document types:
        - bill: Hospital or medical bill with charges
        - discharge_summary: Hospital discharge summary with patient details and treatment
        - id_card: Insurance ID card with policy information
        - prescription: Doctor's prescription
        - lab_report: Laboratory test results
        - unknown: If the document doesn't fit any category
        
        Respond with ONLY the document type (bill, discharge_summary, id_card, prescription, lab_report, or unknown).
        Do not include any other text in your response."""
    
    def _parse_classification_response(self, response: str) -> DocumentType:
        """Parse the LLM response to get the document type."""
        # Clean the response
        doc_type = response.strip().lower()
        
        # Map to our DocumentType enum
        type_mapping = {
            'bill': DocumentType.BILL,
            'discharge': DocumentType.DISCHARGE_SUMMARY,
            'discharge summary': DocumentType.DISCHARGE_SUMMARY,
            'discharge_summary': DocumentType.DISCHARGE_SUMMARY,
            'id': DocumentType.ID_CARD,
            'id card': DocumentType.ID_CARD,
            'id_card': DocumentType.ID_CARD,
            'insurance card': DocumentType.ID_CARD,
            'prescription': DocumentType.PRESCRIPTION,
            'lab': DocumentType.LAB_REPORT,
            'lab report': DocumentType.LAB_REPORT,
            'lab_report': DocumentType.LAB_REPORT,
            'test results': DocumentType.LAB_REPORT,
        }
        
        return type_mapping.get(doc_type, DocumentType.UNKNOWN)
    
    async def process(self, file_path: Path) -> Dict[str, Any]:
        """
        Process a document, classify it, and extract relevant data.
        
        Args:
            file_path: Path to the document to process
            
        Returns:
            Dict containing the document type, extracted data, and processing status
        """
        try:
            # Classify the document
            doc_type = await self.classify_document(file_path)
            
            # Extract text and data based on document type
            text = await self._extract_text(file_path)
            extracted_data = await self.extract_data(text, doc_type, file_path.name)
            
            # Add additional metadata
            extracted_data.update({
                'file_name': file_path.name,
                'file_size': file_path.stat().st_size,
                'processing_date': datetime.now().isoformat(),
                'status': 'processed'
            })
            
            return extracted_data
            
        except Exception as e:
            logger.error(f"Failed to process document {file_path}: {str(e)}")
            return {
                'document_type': DocumentType.UNKNOWN.name,
                'file_name': file_path.name,
                'status': 'error',
                'error': str(e)
            }
            
    async def extract_data(self, text: str, doc_type: DocumentType, filename: str = '') -> Dict[str, Any]:
        """
        Extract structured data from document text based on its type.
        
        Args:
            text: Extracted text from the document
            doc_type: Type of the document
            filename: Original filename (for reference)
            
        Returns:
            Dictionary containing extracted data
        """
        # Initialize result with basic info
        result: ExtractedData = {
            'document_type': doc_type.name,
            'raw_text': text[:5000],  # Store first 5000 chars for reference
        }
        
        try:
            # Extract common fields that might appear in any document
            self._extract_common_fields(text, result)
            
            # Extract type-specific fields
            if doc_type == DocumentType.BILL:
                self._extract_bill_data(text, result)
            elif doc_type == DocumentType.DISCHARGE_SUMMARY:
                self._extract_discharge_summary_data(text, result)
            elif doc_type == DocumentType.ID_CARD:
                self._extract_id_card_data(text, result)
            elif doc_type == DocumentType.PRESCRIPTION:
                self._extract_prescription_data(text, result)
            elif doc_type == DocumentType.LAB_REPORT:
                self._extract_lab_report_data(text, result)
                
        except Exception as e:
            logger.warning(f"Error extracting data from {filename}: {str(e)}")
            result['extraction_errors'] = str(e)
            
        return result
        
    def _extract_common_fields(self, text: str, result: ExtractedData) -> None:
        """Extract fields that are common across document types."""
        # Extract patient name (simple pattern - would need refinement for production)
        name_matches = re.findall(r'(?i)patient(?:\'?s)?[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', text)
        if name_matches:
            result['patient_name'] = name_matches[0].strip()
            
        # Extract dates (simple pattern - would need refinement)
        date_matches = re.findall(r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b', text)
        if date_matches:
            result['date_of_service'] = date_matches[0]
            
        # Extract amounts (simple pattern for demonstration)
        amount_matches = re.findall(r'\$\s*(\d+(?:\.\d{2})?)', text)
        if amount_matches:
            try:
                result['total_amount'] = float(amount_matches[-1])  # Often the last amount is the total
            except (ValueError, IndexError):
                pass
    
    def _extract_bill_data(self, text: str, result: ExtractedData) -> None:
        """Extract data specific to medical bills."""
        # Extract hospital/provider name (simplified)
        hospital_matches = re.findall(r'(?i)hospital|clinic|medical center|healthcare', text)
        if hospital_matches:
            # Look for the hospital name around these keywords
            lines = text.split('\n')
            for i, line in enumerate(lines):
                if any(term in line.lower() for term in ['hospital', 'clinic', 'medical', 'healthcare']):
                    result['hospital_name'] = line.strip()
                    break
                    
        # Extract diagnosis codes (ICD-10 format)
        diag_codes = re.findall(r'\b[A-Z]\d{2}(?:\.\d+)?\b', text)
        if diag_codes:
            result['diagnosis_codes'] = list(set(diag_codes))  # Remove duplicates
            
        # Extract procedure codes (CPT/HCPCS format)
        proc_codes = re.findall(r'\b\d{4}[A-Z]?\b|\b[A-Z]\d{4}\b', text)
        if proc_codes:
            result['procedure_codes'] = list(set(proc_codes))  # Remove duplicates
    
    def _extract_discharge_summary_data(self, text: str, result: ExtractedData) -> None:
        """Extract data specific to discharge summaries."""
        # Discharge summaries might contain similar data to bills
        self._extract_bill_data(text, result)
        
        # Additional discharge-specific fields could be extracted here
        
    def _extract_id_card_data(self, text: str, result: ExtractedData) -> None:
        """Extract data from insurance ID cards."""
        # Look for policy numbers (various formats)
        policy_matches = re.findall(r'(?i)policy(?:\s*#?\s*[:\-]?\s*)([A-Z0-9-]+)', text)
        if policy_matches:
            result['policy_number'] = policy_matches[0]
            
        # Look for insurance provider names
        provider_keywords = ['unitedhealth', 'aetna', 'cigna', 'blue cross', 'medicare', 'medicaid']
        for keyword in provider_keywords:
            if keyword in text.lower():
                result['insurance_provider'] = keyword.title()
                break
    
    def _extract_prescription_data(self, text: str, result: ExtractedData) -> None:
        """Extract data from prescriptions."""
        # Simple extraction - would need to be enhanced for production
        med_matches = re.findall(r'(?i)medication[:\s]+([^\n]+)', text)
        if med_matches:
            result['medications'] = [m.strip() for m in med_matches[0].split(',') if m.strip()]
    
    def _extract_lab_report_data(self, text: str, result: ExtractedData) -> None:
        """Extract data from lab reports."""
        # Look for test results
        test_results = re.findall(r'(?i)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)[\s:]+([\d\.]+)\s*([^\n]+)', text)
        if test_results:
            result['test_results'] = [
                {'test': test.strip(), 'value': value.strip(), 'unit': unit.strip()}
                for test, value, unit in test_results
            ]
