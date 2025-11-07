from pathlib import Path
from typing import Dict, Any, List, Optional, Type, TypeVar, Union, Tuple
import json
import re
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import numpy as np
from datetime import datetime, date
from loguru import logger

from .base_extraction_agent import BaseExtractionAgent
from app.schemas.claim import BillDocument, DocumentType

T = TypeVar('T', bound=BillDocument)

class BillAgent(BaseExtractionAgent[BillDocument]):
    """Agent responsible for processing medical bill documents."""
    
    def __init__(self):
        super().__init__(output_model=BillDocument)
    
    async def _extract_text_from_pdf(self, file_path: Union[str, Path]) -> str:
        """Extract text from PDF, handling both text-based and image-based PDFs."""
        try:
            text = ""
            
            # Open the PDF
            doc = fitz.open(file_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # First try to extract text directly
                page_text = page.get_text()
                
                # If no text or too little text, try OCR
                if not page_text.strip() or len(page_text.strip()) < 50:
                    # Convert PDF page to image
                    pix = page.get_pixmap()
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    
                    # Use Tesseract OCR to extract text
                    page_text = pytesseract.image_to_string(img)
                    logger.info(f"Extracted text from page {page_num + 1} using OCR")
                
                text += f"\n\n{page_text}"
                
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise
    
    async def extract(self, text: Union[str, Path], file_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Extract structured data from medical bill text or file.
        
        Args:
            text: Extracted text from the bill document or path to the file
            file_path: Optional path to the original file (for backward compatibility)
                
        Returns:
            Dictionary containing extracted bill information
        """
        try:
            # If text is a Path, read the file content
            if isinstance(text, (str, Path)) and str(text).endswith('.pdf'):
                text = await self._extract_text_from_pdf(text)
            elif file_path and str(file_path).endswith('.pdf'):
                text = await self._extract_text_from_pdf(file_path)
            
            # If we have a file path but no text, try reading it as a text file
            if not text and file_path and file_path.exists():
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
            
            if not text:
                raise ValueError("No text content found in the document")
                
            logger.info(f"Extracted text: {text[:500]}...")  # Log first 500 chars
            
            # First try to extract common fields with regex
            extracted = self._extract_with_regex(text)
            
            # Then enhance with LLM extraction if needed
            if not all(k in extracted for k in ['total_amount', 'date_of_service']):
                llm_data = await self._extract_with_llm(text)
                # Merge the results, with LLM data taking precedence
                extracted.update(llm_data)
            
            # Ensure required fields are present
            if not all(k in extracted for k in ['total_amount', 'date_of_service']):
                raise ValueError("Missing required fields in bill data")
                
            return extracted
            
        except Exception as e:
            logger.error(f"Error extracting bill data: {str(e)}")
            raise
    
    def _extract_with_regex(self, text: str) -> Dict[str, Any]:
        """Extract common fields using regex patterns."""
        result = {}
        
        # Try to extract using simple format first
        if 'HOSPITAL BILL' in text:
            # Extract patient name
            name_match = re.search(r'Patient Name:\s*([^\n]+)', text, re.IGNORECASE)
            if name_match:
                result['patient_name'] = name_match.group(1).strip()
                
            # Extract date of service
            date_match = re.search(r'Date of Service:\s*([\d-]+)', text, re.IGNORECASE)
            if date_match:
                result['date_of_service'] = self._parse_date(date_match.group(1))
                
            # Extract total amount
            amount_match = re.search(r'Total Amount:\s*\$?(\d+\.\d{2})', text, re.IGNORECASE)
            if amount_match:
                try:
                    result['total_amount'] = float(amount_match.group(1))
                except (ValueError, IndexError):
                    pass
                    
            # Extract line items
            items = []
            for line in text.split('\n'):
                if line.strip().startswith('-'):
                    item_match = re.match(r'-\s*([^:]+):\s*\$?(\d+\.\d{2})', line)
                    if item_match:
                        items.append({
                            'description': item_match.group(1).strip(),
                            'amount': float(item_match.group(2))
                        })
            if items:
                result['items'] = items
                
            # Set a default hospital name if not found
            if 'hospital_name' not in result:
                result['hospital_name'] = 'General Hospital'
                
            return result
                
        # Fall back to the original extraction logic for other formats
        
        # Extract hospital name (look for common hospital name patterns)
        hospital_matches = re.findall(
            r'(?i)(?:hospital|medical center|healthcare|clinic)[\s:]+([^\n]+)', 
            text
        )
        if hospital_matches:
            result['hospital_name'] = hospital_matches[0].strip()
        
        # Extract total amount (look for common total amount patterns)
        amount_matches = re.findall(
            r'(?i)(?:total|amount due|balance)[\s:]*[\$\s]*(\d+(?:\.\d{2})?)', 
            text
        )
        if amount_matches:
            try:
                result['total_amount'] = float(amount_matches[-1])
            except (ValueError, IndexError):
                pass
        
        # Extract date of service (various date formats)
        date_matches = re.findall(
            r'(?i)(?:date of service|service date|date)[\s:]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', 
            text
        )
        if date_matches:
            result['date_of_service'] = self._parse_date(date_matches[0])
            
        # Extract patient information
        name_matches = re.findall(
            r'(?i)patient(?:\'?s)?[\s:]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', 
            text
        )
        if name_matches:
            result['patient_name'] = name_matches[0].strip()
            
        # Extract diagnosis codes (ICD-10 format)
        diag_codes = re.findall(r'\b[A-Z]\d{2}(?:\.\d+)?\b', text)
        if diag_codes:
            result['diagnosis_codes'] = list(set(diag_codes))
            
        # Extract procedure codes (CPT/HCPCS format)
        proc_codes = re.findall(r'\b\d{4}[A-Z]?\b|\b[A-Z]\d{4}\b', text)
        if proc_codes:
            result['procedure_codes'] = list(set(proc_codes))
            
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
        return await self._parse_llm_response(response)
    
    def _create_extraction_prompt(self, text: str) -> str:
        """Create a prompt for extracting structured data from bill text."""
        return f"""You are an expert medical bill processor. Extract the following information from the bill:

Bill Text:
{text}

Extract the following information in JSON format:
1. hospital_name: Name of the hospital or medical facility
2. total_amount: Total amount due (as a number)
3. date_of_service: Date of service in YYYY-MM-DD format
4. patient_name: Name of the patient (if available)
5. patient_id: Patient ID or MRN (if available)
6. diagnosis_codes: List of diagnosis codes (ICD-10 format)
7. procedure_codes: List of procedure codes (CPT/HCPCS format)
8. items: List of items with description, quantity, and amount

Example Output:
{{
  "hospital_name": "General Hospital",
  "total_amount": 1250.75,
  "date_of_service": "2024-04-10",
  "patient_name": "John Doe",
  "patient_id": "MRN123456",
  "diagnosis_codes": ["E11.65", "I10"],
  "procedure_codes": ["99213", "J3423"],
  "items": [
    {{"description": "Doctor Consultation", "quantity": 1, "amount": 250.00}},
    {{"description": "Lab Tests", "quantity": 2, "amount": 500.00}},
    {{"description": "Medication", "quantity": 1, "amount": 500.75}}
  ]
}}

Extracted Data:
"""
    
    async def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response into a structured dictionary."""
        try:
            # Clean the response to ensure it's valid JSON
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)
                
                # Convert date strings to YYYY-MM-DD format
                if 'date_of_service' in data and isinstance(data['date_of_service'], str):
                    try:
                        # Try to parse the date and reformat it
                        parsed_date = self._parse_date(data['date_of_service'])
                        if parsed_date:
                            data['date_of_service'] = parsed_date
                    except (ValueError, TypeError):
                        pass
                        
                return data
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return {}
            
    def _parse_date(self, date_str: str) -> str:
        """Parse date string into YYYY-MM-DD format."""
        if not date_str or not isinstance(date_str, str):
            return ""
            
        # Clean the date string
        date_str = date_str.strip()
        
        try:
            # Try different date formats
            for fmt in ('%Y-%m-%d', '%m/%d/%Y', '%m-%d-%Y', '%d/%m/%Y', '%d-%m-%Y'):
                try:
                    dt = datetime.strptime(date_str, fmt)
                    return dt.strftime('%Y-%m-%d')
                except ValueError:
                    continue
                    
            # If we get here, no format matched
            logger.warning(f"Could not parse date: {date_str}")
            return date_str
            
        except Exception as e:
            logger.warning(f"Error parsing date '{date_str}': {str(e)}")
            return date_str
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
                dt = datetime.strptime(date_str.strip(), fmt)
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                continue
                
        return None
