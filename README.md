# AegisClaim Engine

An AI-powered medical insurance claim processing system that automates document classification, data extraction, and claim validation.

## Architecture & Logic

### Core Components

1. **Document Processing Pipeline**
   - **Document Ingestion**: Accepts multiple document formats (PDF, images, text)
   - **Text Extraction**: Converts documents to text using OCR (for images) or direct text extraction (for PDFs)
   - **Document Classification**: Uses AI to classify documents into categories (bills, ID cards, prescriptions, etc.)
   - **Data Extraction**: Specialized agents extract structured data from each document type
   - **Validation**: Cross-document validation and business rule application
   - **Decision Making**: Automated claim approval/rejection based on extracted data

2. **Agent-Based Architecture**
   - **Base Agent**: Common functionality for all agents (text extraction, LLM calls)
   - **Specialized Agents**: One agent per document type (BillAgent, IDCardAgent, etc.)
   - **Orchestrator**: Coordinates the processing flow between agents

3. **API Layer**
   - RESTful endpoints for claim submission and status checking
   - Asynchronous processing for better scalability
   - Comprehensive error handling and logging

## Features

- **Document Classification**: Automatically classifies medical documents (bills, ID cards, prescriptions, etc.)
- **Data Extraction**: Extracts structured data from various document types using AI
- **Claim Validation**: Validates claim data across multiple documents
- **Automated Decisions**: Makes claim approval/rejection decisions based on configured rules
- **RESTful API**: Easy integration with other systems via HTTP API
- **Asynchronous Processing**: Handles multiple requests efficiently

## AI Tools & Integration

### Google Gemini API Integration
- Used for natural language understanding and information extraction
- Handles document classification and complex data extraction tasks
- Processes both text and image-based documents

### Document Processing
- **PyMuPDF (fitz)**: For PDF text and image extraction
- **Pytesseract**: OCR for image-based documents
- **Pillow**: Image processing for OCR optimization

### How AI is Used
1. **Document Classification**
   - Analyzes document content and filename
   - Classifies into one of: bill, discharge_summary, id_card, prescription, or lab_report

2. **Information Extraction**
   - Extracts structured data from unstructured text
   - Handles variations in document formats
   - Validates and normalizes extracted data

3. **Decision Making**
   - Validates claim requirements
   - Identifies missing or inconsistent information
   - Supports human-in-the-loop verification

## Tech Stack

- **Backend**: FastAPI (Python 3.9+)
- **AI/ML**: Google Gemini API
- **Document Processing**: PyMuPDF, Pytesseract, Pillow, PyPDF2
- **Data Validation**: Pydantic
- **Testing**: Pytest
- **Containerization**: Docker & Docker Compose

## Prerequisites

- Python 3.9 or higher
- Google Gemini API key
- pip (Python package manager)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/aegisclaim-engine.git
   cd aegisclaim-engine
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

## Configuration

Edit the `.env` file with your configuration:

```ini
# Application Settings
ENVIRONMENT=development
DEBUG=True

# API Configuration
API_V1_STR=/api/v1

# File Uploads
UPLOAD_DIR=uploads
MAX_UPLOAD_SIZE=10485760  # 10MB in bytes

# CORS (comma-separated list of allowed origins, or * for all)
BACKEND_CORS_ORIGINS=*

# Google Gemini API
GEMINI_API_KEY=your-gemini-api-key-here
GEMINI_MODEL=gemini-pro

# Rate Limiting
RATE_LIMIT=60  # requests per minute

# Logging
LOG_LEVEL=INFO
```

## Running the Application

### Development Mode

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

### Production Mode

For production, use a production-grade ASGI server like Uvicorn with Gunicorn:

```bash
gunicorn -k uvicorn.workers.UvicornWorker -w 4 -b 0.0.0.0:8000 app.main:app
```

### Using Docker

Build and run with Docker Compose:

```bash
docker-compose up --build
```

## API Documentation

Once the application is running, you can access:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI Schema**: `http://localhost:8000/api/v1/openapi.json`

## Available Endpoints

### Process Claim

- **URL**: `/api/v1/claims/process-claim`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `files`: List of document files (PDF, JPG, PNG)

**Example Request**:

```bash
curl -X 'POST' \
  'http://localhost:8000/api/v1/claims/process-claim' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'files=@bill.pdf;type=application/pdf' \
  -F 'files=@id_card.jpg;type=image/jpeg'
```

**Example Response**:

```json
{
  "documents": [
    {
      "type": "bill",
      "hospital_name": "City General Hospital",
      "total_amount": 1250.75,
      "date_of_service": "2024-04-10",
      "items": [
        {
          "description": "Doctor Consultation",
          "quantity": 1,
          "amount": 250.0
        },
        {
          "description": "Lab Tests",
          "quantity": 2,
          "amount": 500.0
        }
      ]
    },
    {
      "type": "id_card",
      "insurance_provider": "Blue Cross Blue Shield",
      "policy_number": "GP123456789",
      "member_id": "MEMBER12345"
    }
  ],
  "validation": {
    "missing_documents": [],
    "discrepancies": [],
    "is_valid": true
  },
  "decision": {
    "status": "approved",
    "reason": "Claim meets all requirements",
    "amount_approved": 1250.75,
    "amount_rejected": 0.0
  }
}
```

### Health Check

- **URL**: `/health`
- **Method**: `GET`

**Example Response**:

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "environment": "development"
}
```

## AI Prompts

### Document Classification Prompt
```python
"""You are an expert document classifier for medical insurance claims. 
Your task is to classify the following document based on its filename and content.

Filename: {filename}
Document content (first 2000 chars):
{text}

Possible document types:
- bill: Hospital or medical bill with charges
- discharge_summary: Hospital discharge summary with patient details and treatment
- id_card: Insurance ID card with policy information
- prescription: Doctor's prescription
- lab_report: Laboratory test results
- unknown: If the document doesn't fit any category

Respond with ONLY the document type (bill, discharge_summary, id_card, prescription, lab_report, or unknown).
Do not include any other text in your response."""
```

### Medical Bill Extraction Prompt
```python
"""You are an expert medical bill processor. Extract the following information from the bill:

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
{
  "hospital_name": "General Hospital",
  "total_amount": 1250.75,
  "date_of_service": "2024-04-10",
  "patient_name": "John Doe",
  "patient_id": "MRN123456",
  "diagnosis_codes": ["E11.65", "I10"],
  "procedure_codes": ["99213", "J3423"],
  "items": [
    {"description": "Doctor Consultation", "quantity": 1, "amount": 250.00},
    {"description": "Lab Tests", "quantity": 2, "amount": 500.00},
    {"description": "Medication", "quantity": 1, "amount": 500.75}
  ]
}"""
```

### Insurance ID Card Extraction Prompt
```python
"""You are an expert at processing insurance ID cards. Extract the following information from the ID card:

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
{
  "insurance_provider": "Blue Cross Blue Shield",
  "policy_number": "GP123456789",
  "member_id": "MEMBER12345",
  "member_name": "John A. Smith",
  "group_number": "GRP987654",
  "relationship": "self",
  "effective_date": "2024-01-01",
  "expiration_date": "2024-12-31"
}"""
```

### Discharge Summary Extraction Prompt
```python
"""You are an expert at extracting information from hospital discharge summaries. 
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
{
    "patient_name": "Patient's full name",
    "diagnosis": "Primary diagnosis",
    "admission_date": "YYYY-MM-DD",
    "discharge_date": "YYYY-MM-DD",
    "procedures": ["Procedure 1", "Procedure 2", ...],
    "medications": ["Medication 1", "Medication 2", ...],
    "attending_physician": "Dr. Name",
    "facility_name": "Hospital/Clinic Name"
}"""
```

## Testing

Run tests with pytest:

```bash
pytest tests/
```

## Deployment

### Docker

Build the Docker image:

```bash
docker build -t aegisclaim-engine .
```

Run the container:

```bash
docker run -d -p 8000:8000 --env-file .env aegisclaim-engine
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Support

For support, please open an issue in the GitHub repository.

## Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/)
- [Google Gemini](https://ai.google.dev/)
- [Pydantic](https://pydantic-docs.helpmanual.io/)
- [Uvicorn](https://www.uvicorn.org/)
