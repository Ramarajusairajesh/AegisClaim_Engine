from pathlib import Path
from typing import Dict, Any, List, Optional, Type, TypeVar
import json
import re
from datetime import datetime, date
from loguru import logger

from .base_extraction_agent import BaseExtractionAgent, AgentError
from app.schemas.claim import DischargeSummaryDocument, DocumentType

T = TypeVar('T', bound=DischargeSummaryDocument)

class DischargeAgent(BaseExtractionAgent[DischargeSummaryDocument]):
    """Agent responsible for processing hospital discharge summary documents."""
    
    def __init__(self):
        super().__init__(output_model=DischargeSummaryDocument)
    
    async def extract(self, text: str, file_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Extract structured data from discharge summary text.
        
        Args:
            text: Extracted text from the discharge summary
            file_path: Optional path to the original file
            
        Returns:
            Dictionary containing extracted discharge summary information
        """
        try:
            # First try to extract common fields with regex
            extracted = self._extract_with_regex(text)
            
            # Then enhance with LLM extraction
            llm_data = await self._extract_with_llm(text)
            
            # Merge the results, with LLM data taking precedence
            extracted.update(llm_data)
            
            # Ensure required fields are present
            required_fields = ['patient_name', 'diagnosis', 'admission_date', 'discharge_date']
            if not all(k in extracted for k in required_fields):
                missing = [f for f in required_fields if f not in extracted]
                raise ValueError(f"Missing required fields in discharge summary: {', '.join(missing)}")
                
            return extracted
            
        except Exception as e:
            logger.error(f"Error extracting discharge summary data: {str(e)}")
            raise
    
    def _extract_with_regex(self, text: str) -> Dict[str, Any]:
        """Extract common fields using regex patterns."""
        result = {}
        
        # Extract patient name
        name_matches = re.findall(
            r'(?i)patient(?:\'?s)?[\s:]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', 
            text
        )
        if name_matches:
            result['patient_name'] = name_matches[0].strip()
        
        # Extract diagnosis (look for common diagnosis patterns)
        diag_matches = re.findall(
            r'(?i)(?:diagnosis|diagnoses|dx)[\s:]+([^\n]+?)(?=\n\s*\w|$)', 
            text, 
            re.DOTALL
        )
        if diag_matches:
            result['diagnosis'] = diag_matches[0].strip()
        
        # Extract admission and discharge dates
        date_patterns = [
            (r'(?i)admi(?:ssion)?[\s:]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', 'admission_date'),
            (r'(?i)discharge[\s:]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', 'discharge_date'),
        ]
        
        for pattern, field in date_patterns:
            matches = re.search(pattern, text)
            if matches:
                result[field] = self._parse_date(matches.group(1))
        
        # Extract procedures
        proc_matches = re.findall(
            r'(?i)procedure(?:s)?[\s:]+([^\n]+?)(?=\n\s*\w|$)', 
            text, 
            re.DOTALL
        )
        if proc_matches:
            # Split procedures by common separators
            procedures = re.split(r'[,\n]', proc_matches[0])
            result['procedures'] = [p.strip() for p in procedures if p.strip()]
        
        # Extract medications
        med_matches = re.findall(
            r'(?i)medication(?:s)?[\s:]+([^\n]+?)(?=\n\s*\w|$)', 
            text, 
            re.DOTALL
        )
        if med_matches:
            # Split medications by common separators
            meds = re.split(r'[,\n]', med_matches[0])
            result['medications'] = [m.strip() for m in meds if m.strip()]
            
        # Extract physician name
        doc_matches = re.findall(
            r'(?i)(?:attending|primary)\s+physician[\s:]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
            text
        )
        if doc_matches:
            result['attending_physician'] = doc_matches[0].strip()
            
        # Extract facility name
        facility_matches = re.findall(
            r'(?i)(?:facility|hospital)[\s:]+([^\n]+)',
            text
        )
        if facility_matches:
            result['facility_name'] = facility_matches[0].strip()
            
        return result
    
    async def _extract_with_llm(self, text: str) -> Dict[str, Any]:
        """Extract structured data using LLM for more complex cases."""
        prompt = self._create_extraction_prompt(text[:4000])  # Use first 4000 chars for extraction
        
        # Get response from LLM
        response = await self._call_llm(
            prompt,
            generation_config={
                "temperature": 0.1,  # Lower temperature for more deterministic output
                "max_output_tokens": 1000,
            }
        )
        
        # Parse the response
        return self._parse_extraction_response(response)
    
    def _parse_date(self, date_str: Optional[str]) -> Optional[date]:
        """Parse date string into date object."""
        if not date_str:
            return None
        try:
            return datetime.strptime(date_str, "%Y-%m-%d").date()
        except (ValueError, TypeError):
            logger.warning(f"Invalid date format: {date_str}")
            return None
    
    def _create_extraction_prompt(self, text: str) -> str:
        """Create a prompt for extracting discharge summary information."""
        return f"""You are an expert at extracting information from hospital discharge summaries. 
Extract the following information from the discharge summary below:

1. Patient Name
2. Primary Diagnosis
3. Admission Date (in YYYY-MM-DD format)
4. Discharge Date (in YYYY-MM-DD format)
5. List of medical procedures performed
6. List of prescribed medications
7. Attending Physician Name (if available)
8. Facility/Hospital Name

Discharge Summary Content:
{text}

Respond with a JSON object in this exact format:
{{
    "patient_name": "Patient's full name",
    "diagnosis": "Primary diagnosis",
    "admission_date": "YYYY-MM-DD",
    "discharge_date": "YYYY-MM-DD",
    "procedures": ["Procedure 1", "Procedure 2", ...],
    "medications": ["Medication 1", "Medication 2", ...],
    "attending_physician": "Dr. Name",
    "facility_name": "Hospital/Clinic Name"
}}

If any information is not found, use null for that field.
"""
    
    def _parse_extraction_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response into a structured dictionary."""
        try:
            # Clean the response to extract JSON
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON object found in response")
                
            json_str = response[json_start:json_end]
            data = json.loads(json_str)
            
            # Ensure all required fields exist
            for field in ["patient_name", "diagnosis", "procedures", "medications"]:
                if field not in data:
                    data[field] = None if field != "procedures" and field != "medications" else []
            
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from response: {e}")
            logger.debug(f"Response was: {response}")
            raise AgentError("Failed to parse discharge summary information from document")
        except Exception as e:
            logger.error(f"Error parsing discharge summary data: {str(e)}")
            raise AgentError(f"Failed to process discharge summary: {str(e)}")
