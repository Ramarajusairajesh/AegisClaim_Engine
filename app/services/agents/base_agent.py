from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional
import google.generativeai as genai
from loguru import logger

from app.config import settings

class BaseAgent(ABC):
    """Base class for all AI agents in the system."""
    
    def __init__(self):
        self._init_gemini()
    
    def _init_gemini(self):
        """Initialize the Gemini API client."""
        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
        
        # Configure the Gemini API with the correct API key and settings
        genai.configure(api_key=settings.GEMINI_API_KEY)
        
        # Use a model that's known to be supported
        # Using gemini-pro-latest which is listed in the available models
        self.model_name = "gemini-pro-latest"  # Using the latest stable model
        
        # Initialize the model with generation config
        generation_config = {
            "temperature": 0.7,
            "top_p": 1,
            "top_k": 40,
            "max_output_tokens": 2048,
        }
        
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
        ]
        
        # Initialize the model
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        logger.info(f"Initialized Gemini model: {self.model_name}")
    
    @abstractmethod
    async def process(self, file_path: Path) -> Any:
        """Process a document and return structured data."""
        pass
    
    async def _extract_text(self, file_path: Path) -> str:
        """Extract text from a document file.
        
        Args:
            file_path: Path to the file to extract text from
            
        Returns:
            Extracted text from the file
            
        Raises:
            AgentError: If the file cannot be read or is not a supported format
        """
        try:
            # Check if file exists and is readable
            if not file_path.exists():
                raise AgentError(f"File not found: {file_path}")
                
            # Handle PDF files
            if file_path.suffix.lower() == '.pdf':
                import PyPDF2
                text_parts = []
                try:
                    with open(file_path, 'rb') as f:
                        reader = PyPDF2.PdfReader(f)
                        for page in reader.pages:
                            text_parts.append(page.extract_text() or "")
                    return "\n".join(text_parts)
                except Exception as e:
                    raise AgentError(f"Failed to extract text from PDF {file_path}: {str(e)}")
            
            # Handle text files
            elif file_path.suffix.lower() in ['.txt', '.md', '.csv']:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    return f.read()
                    
            else:
                raise AgentError(f"Unsupported file format: {file_path.suffix}")
                
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {str(e)}")
            raise AgentError(f"Failed to extract text: {str(e)}")
    
    async def _call_llm(self, prompt: str, **kwargs) -> str:
        """Make a call to the LLM with the given prompt."""
        try:
            response = await self.model.generate_content_async(prompt, **kwargs)
            return response.text
        except Exception as e:
            logger.error(f"Error calling Gemini API: {str(e)}")
            raise

class AgentError(Exception):
    """Base exception for agent-related errors."""
    pass
