import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from app.schemas.document import DocumentType
from app.services.agents.classifier_agent import ClassifierAgent

@pytest.fixture
def classifier_agent():
    return ClassifierAgent()

@pytest.mark.asyncio
async def test_classify_document_success(classifier_agent):
    # Mock the _extract_text and _call_llm methods
    classifier_agent._extract_text = AsyncMock(return_value="Sample hospital bill content")
    classifier_agent._call_llm = AsyncMock(return_value="bill")  # Simulate LLM response
    
    # Create a test file
    test_file = Path("test_bill.pdf")
    
    # Test the classification
    result = await classifier_agent.classify_document(test_file)
    
    # Verify the result
    assert result == DocumentType.BILL
    classifier_agent._extract_text.assert_awaited_once_with(test_file)
    classifier_agent._call_llm.assert_awaited_once()

@pytest.mark.asyncio
async def test_classify_document_unknown_type(classifier_agent):
    classifier_agent._extract_text = AsyncMock(return_value="Some random content")
    classifier_agent._call_llm = AsyncMock(return_value="unknown")
    
    test_file = Path("unknown.txt")
    result = await classifier_agent.classify_document(test_file)
    
    assert result == DocumentType.UNKNOWN

@pytest.mark.asyncio
async def test_classify_document_extraction_error(classifier_agent):
    classifier_agent._extract_text = AsyncMock(side_effect=Exception("Failed to extract text"))
    
    test_file = Path("error.pdf")
    result = await classifier_agent.classify_document(test_file)
    
    assert result == DocumentType.UNKNOWN

def test_parse_classification_response(classifier_agent):
    # Test various response formats
    assert classifier_agent._parse_classification_response("bill") == "bill"
    assert classifier_agent._parse_classification_response("  BILL  ") == "bill"
    assert classifier_agent._parse_classification_response("discharge summary") == "discharge_summary"
    assert classifier_agent._parse_classification_response("id card") == "id_card"
    assert classifier_agent._parse_classification_response("invalid") == "unknown"
