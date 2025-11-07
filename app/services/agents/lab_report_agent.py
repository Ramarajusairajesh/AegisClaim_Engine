from pathlib import Path
from typing import Dict, Any, List, Optional, Type, TypeVar, Union
import json
import re
from datetime import datetime, date
from loguru import logger

from .base_extraction_agent import BaseExtractionAgent
from app.schemas.claim import LabReportDocument, DocumentType

T = TypeVar('T', bound=LabReportDocument)

class LabReportAgent(BaseExtractionAgent[LabReportDocument]):
    """Agent responsible for processing laboratory test reports."""
    
    def __init__(self):
        super().__init__(output_model=LabReportDocument)
    
    async def extract(self, text: str, file_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Extract structured data from lab report text.
        
        Args:
            text: Extracted text from the lab report
            file_path: Optional path to the original file
            
        Returns:
            Dictionary containing extracted lab report information
        """
        try:
            # First try to extract common fields with regex
            extracted = self._extract_with_regex(text)
            
            # Then enhance with LLM extraction
            llm_data = await self._extract_with_llm(text)
            
            # Merge the results, with LLM data taking precedence
            extracted.update(llm_data)
            
            # Ensure required fields are present
            required_fields = ['patient_name', 'test_results', 'date_collected']
            if not all(k in extracted for k in required_fields):
                missing = [f for f in required_fields if f not in extracted]
                raise ValueError(f"Missing required fields in lab report: {', '.join(missing)}")
                
            # Set default reported date if not present
            if 'date_reported' not in extracted:
                extracted['date_reported'] = datetime.now().strftime('%Y-%m-%d')
                
            return extracted
            
        except Exception as e:
            logger.error(f"Error extracting lab report data: {str(e)}")
            raise
    
    def _extract_with_regex(self, text: str) -> Dict[str, Any]:
        """Extract common fields using regex patterns."""
        result = {
            'test_results': []
        }
        
        # Extract patient name
        name_matches = re.findall(
            r'(?i)patient(?:\'?s)?[\s:]+([A-Z][a-z]+(?:\s+[A-Z][a-z\.]+)+)', 
            text
        )
        if name_matches:
            result['patient_name'] = name_matches[0].strip()
        
        # Extract patient ID
        id_matches = re.findall(
            r'(?i)(?:patient[\s-]?id|mrn|medical[\s-]?record[\s-]?number)[\s:]+([A-Z0-9-]+)',
            text
        )
        if id_matches:
            result['patient_id'] = id_matches[0].strip()
        
        # Extract dates (collected and reported)
        date_patterns = [
            (r'(?i)(?:collection|collected|specimen)[\s:]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', 'date_collected'),
            (r'(?i)(?:reported|result|completed)[\s:]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', 'date_reported'),
        ]
        
        for pattern, field in date_patterns:
            matches = re.search(pattern, text)
            if matches:
                parsed_date = self._parse_date(matches.group(1))
                if parsed_date:
                    result[field] = parsed_date
        
        # Extract lab name
        lab_matches = re.findall(
            r'(?i)(?:laboratory|lab|facility)[\s:]+([^\n]+)',
            text
        )
        if lab_matches:
            result['lab_name'] = lab_matches[0].strip()
        
        # Extract ordering physician
        doc_matches = re.findall(
            r'(?i)(?:ordering[\s-]?physician|ordering[\s-]?provider|doctor)[\s:]+([A-Z][a-z]+(?:\s+[A-Z][a-z\.]+)+)',
            text
        )
        if doc_matches:
            result['ordering_physician'] = doc_matches[0].strip()
        
        # Extract test results (simple pattern, will be enhanced by LLM)
        # Look for common test result patterns (e.g., "Glucose: 95 mg/dL")
        test_patterns = [
            (r'([A-Za-z\s]+)[\s:]+([\d\.]+)[\s]*([^\n\d]+)', 'value_unit'),
            (r'([A-Za-z\s]+)[\s]*\(([^)]+)\)[\s:]*([\d\.]+)', 'name_reference_value'),
        ]
        
        for pattern, pattern_type in test_patterns:
            for match in re.finditer(pattern, text):
                try:
                    if pattern_type == 'value_unit':
                        test_name = match.group(1).strip()
                        value = match.group(2).strip()
                        unit = match.group(3).strip()
                        result['test_results'].append({
                            'test_name': test_name,
                            'result': value,
                            'unit': unit,
                            'reference_range': '',
                            'flag': ''
                        })
                    elif pattern_type == 'name_reference_value':
                        test_name = match.group(1).strip()
                        reference = match.group(2).strip()
                        value = match.group(3).strip()
                        result['test_results'].append({
                            'test_name': test_name,
                            'result': value,
                            'reference_range': reference,
                            'unit': '',
                            'flag': ''
                        })
                except (IndexError, ValueError):
                    continue
        
        return result
    
    async def _extract_with_llm(self, text: str) -> Dict[str, Any]:
        """Extract structured data using LLM for more complex cases."""
        prompt = self._create_extraction_prompt(text[:4000])  # Use first 4000 chars for extraction
        
        # Get response from LLM
        response = await self._call_llm(
            prompt,
            generation_config={
                "temperature": 0.1,  # Lower temperature for more deterministic output
                "max_output_tokens": 2000,
            }
        )
        
        # Parse the response
        return await self._parse_llm_response(response)
    
    def _create_extraction_prompt(self, text: str) -> str:
        """Create a prompt for extracting structured data from lab report text."""
        return f"""You are an expert at processing laboratory test reports. Extract the following information from the lab report:

Lab Report Text:
{text}

Extract the following information in JSON format:
1. patient_name: Full name of the patient
2. patient_id: Patient ID or MRN (if available)
3. date_collected: Date the specimen was collected (YYYY-MM-DD)
4. date_reported: Date the results were reported (YYYY-MM-DD)
5. lab_name: Name of the laboratory
6. ordering_physician: Name of the ordering physician (if available)
7. test_results: List of test results, each with:
   - test_name: Name of the test
   - result: Test result value
   - unit: Unit of measurement
   - reference_range: Normal reference range
   - flag: Any flags (e.g., H for high, L for low, N for normal)
   - status: Test status (e.g., Final, Preliminary, Corrected)
8. interpretation: Any interpretation or comments from the lab
9. lab_notes: Any additional notes from the lab

Example Output:
{{
  "patient_name": "John A. Smith",
  "patient_id": "MRN123456",
  "date_collected": "2024-04-10",
  "date_reported": "2024-04-11",
  "lab_name": "Quest Diagnostics",
  "ordering_physician": "Dr. Sarah Johnson",
  "test_results": [
    {{
      "test_name": "Glucose",
      "result": "95",
      "unit": "mg/dL",
      "reference_range": "70-99 mg/dL",
      "flag": "N",
      "status": "Final"
    }},
    {{
      "test_name": "Hemoglobin A1c",
      "result": "5.4",
      "unit": "%",
      "reference_range": "<5.7% (Normal)",
      "flag": "N",
      "status": "Final"
    }},
    {{
      "test_name": "LDL Cholesterol",
      "result": "130",
      "unit": "mg/dL",
      "reference_range": "<100 mg/dL (Optimal)",
      "flag": "H",
      "status": "Final"
    }}
  ],
  "interpretation": "Fasting glucose and A1c within normal limits. Elevated LDL cholesterol noted.",
  "lab_notes": "Fasting sample. Results reviewed and verified by laboratory director."
}}

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
            '%m/%d/%y',    # 04/10/24
            '%m-%d-%y',    # 04-10-24
            '%d/%m/%y',    # 10/04/24 (international)
            '%d-%m-%y',    # 10-04-24 (international)
        ]
        
        for fmt in date_formats:
            try:
                dt = datetime.strptime(str(date_str).strip(), fmt)
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                continue
                
        return None
