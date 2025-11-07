from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import logging
from typing import List
from pathlib import Path

from app.api.endpoints import claims
from app.core.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from app.schemas.document import (
    BillDocument,
    DischargeSummaryDocument,
    IDCardDocument,
    DocumentType,
    ClaimResponse,
    ClaimDecision,
    ValidationResult
)
from .services.document_processor import DocumentProcessor
from .services.agents.classifier_agent import ClassifierAgent
from .services.agents.bill_agent import BillAgent
from .services.agents.discharge_agent import DischargeAgent
from .utils.logging import setup_logging

# Set up logging
setup_logging()

def create_application() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="AegisClaim Engine",
        description="AI-powered medical insurance claim processing system",
        version="1.0.0",
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # Configure CORS
    if settings.BACKEND_CORS_ORIGINS:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    else:
        # Allow all origins in development
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Initialize agents
    classifier_agent = ClassifierAgent()
    bill_agent = BillAgent()
    discharge_agent = DischargeAgent()
    document_processor = DocumentProcessor(
        classifier_agent=classifier_agent,
        bill_agent=bill_agent,
        discharge_agent=discharge_agent,
    )

    # Include API routers
    app.include_router(
        claims.router,
        prefix=f"{settings.API_V1_STR}/claims",
        tags=["claims"]
    )

    # Health check endpoint
    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "version": "1.0.0",
            "environment": settings.ENVIRONMENT
        }
    
    # Process claim endpoint
    @app.post("/process-claim", response_model=ClaimResponse)
    async def process_claim(
        files: List[UploadFile] = File(...),
    ):
        """
        Process medical insurance claim documents.
        
        This endpoint accepts multiple PDF files, classifies them, extracts relevant information,
        and returns a structured claim decision.
        """
        try:
            # Validate files
            if not files:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No files provided"
                )
            
            # Process documents
            result = await document_processor.process_documents(files)
        
            return result
        
        except HTTPException as he:
            logger.error(f"HTTP error processing claim: {str(he)}")
            raise he
        except Exception as e:
            logger.error(f"Error processing claim: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error processing claim: {str(e)}"
            )

    # Log application startup
    @app.on_event("startup")
    async def startup_event():
        logger.info("Starting AegisClaim Engine...")
        logger.info(f"Environment: {settings.ENVIRONMENT}")
        logger.info(f"Debug mode: {settings.DEBUG}")
        
        # Ensure upload directory exists
        upload_dir = Path(settings.UPLOAD_DIR)
        upload_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Upload directory: {upload_dir.absolute()}")
    
    return app

# Create the FastAPI application
app = create_application()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)