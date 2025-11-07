from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, List, TypeVar, Generic, Type
from pydantic import BaseModel, ValidationError
from loguru import logger


class AgentError(Exception):
    """Base exception for agent-related errors."""
    pass

T = TypeVar('T', bound=BaseModel)

class BaseExtractionAgent(ABC, Generic[T]):
    """Base class for all extraction agents."""
    
    def __init__(self, output_model: Type[T]):
        """Initialize the extraction agent with an output model.
        
        Args:
            output_model: Pydantic model that defines the expected output structure
        """
        self.output_model = output_model
    
    @abstractmethod
    async def extract(self, text: str, file_path: Optional[Path] = None) -> Dict[str, Any]:
        """Extract structured data from text.
        
        Args:
            text: Extracted text from the document
            file_path: Path to the original file (optional)
            
        Returns:
            Dictionary containing the extracted data
        """
        pass
    
    async def validate(self, data: Dict[str, Any]) -> T:
        """Validate the extracted data against the output model.
        
        Args:
            data: Extracted data to validate
            
        Returns:
            Validated data as an instance of the output model
            
        Raises:
            ValidationError: If the data doesn't match the output model
        """
        try:
            return self.output_model(**data)
        except ValidationError as e:
            logger.error(f"Validation error: {e}")
            raise
    
    async def process(self, text: str, file_path: Optional[Path] = None) -> T:
        """Process the document text and return validated data.
        
        Args:
            text: Extracted text from the document
            file_path: Path to the original file (optional)
            
        Returns:
            Validated data as an instance of the output model
        """
        extracted_data = await self.extract(text, file_path)
        return await self.validate(extracted_data)
