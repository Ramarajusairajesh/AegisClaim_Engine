"""
Services module for the AegisClaim Engine.

This package contains the business logic and service layer components
for processing insurance claims.
"""

# Import key components for easier access
from .document_processor import DocumentProcessor

__all__ = [
    'DocumentProcessor',
]
