from pathlib import Path
from typing import Dict, Any, List, Optional, Type, TypeVar
import json
import re
from datetime import datetime, date
from loguru import logger

from .base_extraction_agent import BaseExtractionAgent
from app.schemas.claim import IdCardDocument, DocumentType

T = TypeVar('T', bound=IdCardDocument)

class IdCardAgent(BaseExtractionAgent[IdCardDocument]):
    """Agent responsible for processing insurance ID card documents."""
    
    def __init__(self):
        super().__init__(output_model=IdCardDocument)
    
    async def extract(self, text: str, file_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Extract structured data from insurance ID card text.
        
        Args:
            text: Extracted text from the ID card
            file_path: Optional path to the original file
            
        Returns:
            Dictionary containing extracted ID card information
        """
        try:
            # First try to extract common fields with regex
            extracted = self._extract_with_regex(text)
            
            # Then enhance with LLM extraction
            llm_data = await self._extract_with_llm(text)
            
            # Merge the results, with LLM data taking precedence
            extracted.update(llm_data)
            
            # Ensure required fields are present
            required_fields = ['insurance_provider', 'policy_number', 'member_id', 'member_name']
            if not all(k in extracted for k in required_fields):
                missing = [f for f in required_fields if f not in extracted]
                raise ValueError(f"Missing required fields in ID card: {', '.join(missing)}")
                
            return extracted
            
        except Exception as e:
            logger.error(f"Error extracting ID card data: {str(e)}")
            raise
    
    def _extract_with_regex(self, text: str) -> Dict[str, Any]:
        """Extract common fields using regex patterns."""
        result = {}
        
        # Extract insurance provider (common providers)
        provider_keywords = {
            'unitedhealth': 'UnitedHealthcare',
            'aetna': 'Aetna',
            'cigna': 'Cigna',
            'blue cross': 'Blue Cross Blue Shield',
            'blue shield': 'Blue Cross Blue Shield',
            'bcbs': 'Blue Cross Blue Shield',
            'kaiser': 'Kaiser Permanente',
            'humana': 'Humana',
            'medicare': 'Medicare',
            'medicaid': 'Medicaid',
        }
        
        # Look for provider names in the text
        text_lower = text.lower()
        for keyword, provider in provider_keywords.items():
            if keyword in text_lower:
                result['insurance_provider'] = provider
                break
        
        # Extract policy number (various formats)
        policy_matches = re.findall(
            r'(?i)(?:policy|id|number|#)[\s:]*([A-Z0-9-]+)', 
            text
        )
        if policy_matches:
            result['policy_number'] = policy_matches[0].strip()
        
        # Extract member ID (various formats)
        member_id_matches = re.findall(
            r'(?i)(?:member|id|subscriber)[\s:]*([A-Z0-9-]+)', 
            text
        )
        if member_id_matches:
            result['member_id'] = member_id_matches[0].strip()
        
        # Extract member name (look for name after member/subscriber)
        name_matches = re.findall(
            r'(?i)(?:member|subscriber|name)[\s:]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
            text
        )
        if name_matches:
            result['member_name'] = name_matches[0].strip()
        
        # Extract group number (if present)
        group_matches = re.findall(
            r'(?i)(?:group|grp)[\s:]*([A-Z0-9-]+)',
            text
        )
        if group_matches:
            result['group_number'] = group_matches[0].strip()
        
        # Extract dates (effective and expiration)
        date_patterns = [
            (r'(?i)(?:effective|eff\.?)[\s:]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', 'effective_date'),
            (r'(?i)(?:expiration|exp\.?|expires)[\s:]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', 'expiration_date'),
        ]
        
        for pattern, field in date_patterns:
            matches = re.search(pattern, text)
            if matches:
                parsed_date = self._parse_date(matches.group(1))
                if parsed_date:
                    result[field] = parsed_date
        
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
        """Create a prompt for extracting structured data from ID card text."""
        return f"""You are an expert at processing insurance ID cards. Extract the following information from the ID card:

ID Card Text:
{text}

Extract the following information in JSON format:
1. insurance_provider: Name of the insurance company
2. policy_number: Policy or group policy number
3. member_id: Member/Subscriber ID
4. member_name: Name of the insured member
5. group_number: Group number (if applicable)
6. relationship: Relationship to primary policyholder (self, spouse, child, etc.)
7. effective_date: Coverage start date (YYYY-MM-DD)
8. expiration_date: Coverage end date (YYYY-MM-DD, if available)

Example Output:
{{
  "insurance_provider": "Blue Cross Blue Shield",
  "policy_number": "GP123456789",
  "member_id": "MEMBER12345",
  "member_name": "John A. Smith",
  "group_number": "GRP987654",
  "relationship": "self",
  "effective_date": "2024-01-01",
  "expiration_date": "2024-12-31"
}}

Extracted Data:"""
    
    def _parse_date(self, date_str: str) -> Optional[str]:
        """Parse a date string into YYYY-MM-DD format."""
        if not date_str:
            return None
            
        # Common date formats to try
        date_formats = [
            '%Y-%m-%d',    # 2024-01-01
            '%m/%d/%Y',    # 01/01/2024
            '%m-%d-%Y',    # 01-01-2024
            '%d/%m/%Y',    # 01/01/2024 (international)
            '%d-%m-%Y',    # 01-01-2024 (international)
            '%m/%d/%y',    # 01/01/24
            '%m-%d-%y',    # 01-01-24
            '%Y/%m/%d',    # 2024/01/01
            '%b %d, %Y',   # Jan 1, 2024
            '%B %d, %Y',   # January 1, 2024
        ]
        
        for fmt in date_formats:
            try:
                dt = datetime.strptime(str(date_str).strip(), fmt)
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                continue
                
        return None
