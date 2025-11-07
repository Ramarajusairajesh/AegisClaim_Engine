from pathlib import Path
from typing import Dict, Any, List, Optional, Type, TypeVar
import json
import re
from datetime import datetime, date
from loguru import logger

from .base_extraction_agent import BaseExtractionAgent
from app.schemas.claim import PrescriptionDocument, DocumentType

T = TypeVar('T', bound=PrescriptionDocument)

class PrescriptionAgent(BaseExtractionAgent[PrescriptionDocument]):
    """Agent responsible for processing prescription documents."""
    
    def __init__(self):
        super().__init__(output_model=PrescriptionDocument)
    
    async def extract(self, text: str, file_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Extract structured data from prescription text.
        
        Args:
            text: Extracted text from the prescription
            file_path: Optional path to the original file
            
        Returns:
            Dictionary containing extracted prescription information
        """
        try:
            # First try to extract common fields with regex
            extracted = self._extract_with_regex(text)
            
            # Then enhance with LLM extraction
            llm_data = await self._extract_with_llm(text)
            
            # Merge the results, with LLM data taking precedence
            extracted.update(llm_data)
            
            # Ensure required fields are present
            required_fields = ['patient_name', 'date_prescribed', 'medications']
            if not all(k in extracted for k in required_fields):
                missing = [f for f in required_fields if f not in extracted]
                raise ValueError(f"Missing required fields in prescription: {', '.join(missing)}")
                
            return extracted
            
        except Exception as e:
            logger.error(f"Error extracting prescription data: {str(e)}")
            raise
    
    def _extract_with_regex(self, text: str) -> Dict[str, Any]:
        """Extract common fields using regex patterns."""
        result = {
            'medications': []
        }
        
        # Extract patient name
        name_matches = re.findall(
            r'(?i)patient(?:\'?s)?[\s:]+([A-Z][a-z]+(?:\s+[A-Z][a-z\.]+)+)', 
            text
        )
        if name_matches:
            result['patient_name'] = name_matches[0].strip()
        
        # Extract date prescribed
        date_matches = re.findall(
            r'(?i)(?:date|prescribed|rx date)[\s:]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', 
            text
        )
        if date_matches:
            result['date_prescribed'] = self._parse_date(date_matches[0])
        
        # Extract prescriber information
        doc_matches = re.findall(
            r'(?i)(?:prescriber|physician|provider|doctor)[\s:]+([A-Z][a-z]+(?:\s+[A-Z][a-z\.]+)+)',
            text
        )
        if doc_matches:
            result['prescriber_name'] = doc_matches[0].strip()
        
        # Extract prescriber license
        license_matches = re.findall(
            r'(?i)(?:license|lic\.?|deap?)[\s:]*([A-Z0-9]+)',
            text
        )
        if license_matches:
            result['prescriber_license'] = license_matches[0].strip()
        
        # Extract medications (simple pattern, will be enhanced by LLM)
        med_sections = re.findall(
            r'(?i)(?:medication|rx|drug|prescription)[\s:]+([^\n]+?)(?=\n\s*\w|$)', 
            text, 
            re.DOTALL
        )
        
        if med_sections:
            # Simple extraction - will be enhanced by LLM
            med_lines = [line.strip() for line in med_sections[0].split('\n') if line.strip()]
            for line in med_lines:
                if line and len(line) > 3:  # Basic validation
                    result['medications'].append({
                        'name': line,
                        'dosage': '',
                        'frequency': '',
                        'instructions': ''
                    })
        
        return result
    
    async def _extract_with_llm(self, text: str) -> Dict[str, Any]:
        """Extract structured data using LLM for more complex cases."""
        prompt = self._create_extraction_prompt(text[:4000])  # Use first 4000 chars for extraction
        
        # Get response from LLM
        response = await self._call_llm(
            prompt,
            generation_config={
                "temperature": 0.1,  # Lower temperature for more deterministic output
                "max_output_tokens": 1500,
            }
        )
        
        # Parse the response
        return await self._parse_llm_response(response)
    
    def _create_extraction_prompt(self, text: str) -> str:
        """Create a prompt for extracting structured data from prescription text."""
        example_output = """{
  "patient_name": "John A. Smith",
  "date_prescribed": "2024-04-10",
  "prescriber_name": "Dr. Sarah Johnson",
  "prescriber_license": "MD12345678",
  "medications": [
    {
      "name": "Lisinopril",
      "strength": "10mg",
      "form": "tablet",
      "quantity": 30,
      "refills": 3,
      "instructions": "Take 1 tablet by mouth daily for high blood pressure",
      "ndc": "12345-0678-90"
    },
    {
      "name": "Metformin",
      "strength": "500mg",
      "form": "tablet",
      "quantity": 60,
      "refills": 3,
      "instructions": "Take 1 tablet by mouth twice daily with meals",
      "ndc": "54321-1234-56"
    }
  ],
  "instructions": "Take medications as directed. Follow up in 3 months.",
  "pharmacy_notes": "May cause dizziness. Avoid alcohol."
}"""
        
        return f"""You are an expert at processing medical prescriptions. Extract the following information from the prescription:

Prescription Text:
{text}

Extract the following information in JSON format:
1. patient_name: Full name of the patient
2. date_prescribed: Date the prescription was written (YYYY-MM-DD)
3. prescriber_name: Name of the prescribing doctor
4. prescriber_license: License number of the prescriber (if available)
5. medications: List of prescribed medications, each with:
   - name: Name of the medication
   - strength: Dosage strength (e.g., "500mg", "10mg/5ml")
   - form: Form of the medication (e.g., "tablet", "capsule", "liquid")
   - quantity: Number of units prescribed
   - refills: Number of refills allowed
   - instructions: How to take the medication (e.g., "Take 1 tablet by mouth daily")
   - ndc: National Drug Code (if available)
6. instructions: General instructions for the patient
7. pharmacy_notes: Any notes for the pharmacy

Example Output:
{example_output}

Extracted Data:"""
    
    def _parse_date(self, date_str: str) -> Optional[str]:
        """Parse a date string into YYYY-MM-DD format."""
        if not date_str:
            return None
            
        # Common date formats to try
        date_formats = [
            '%Y-%m-%d',    # 2024-04-10
            '%m/%d/%Y',    # 04/10/2024
            '%m-%d-%Y',    # 04-10-2024
            '%d/%m/%Y',    # 10/04/2024 (international)
            '%d-%m-%Y',    # 10-04-2024 (international)
            '%Y/%m/%d',    # 2024/04/10
            '%b %d, %Y',   # Apr 10, 2024
            '%B %d, %Y',   # April 10, 2024
        ]
        
        for fmt in date_formats:
            try:
                dt = datetime.strptime(str(date_str).strip(), fmt)
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                continue
                
        return None
