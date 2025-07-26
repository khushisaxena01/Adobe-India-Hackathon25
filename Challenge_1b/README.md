# Challenge 1b: Multi-Collection PDF Analysis

## Overview
Advanced multi-collection PDF analysis solution that extracts and prioritizes relevant content from document collections based on specific user personas and job-to-be-done requirements. Built for Adobe India Hackathon 2025 Challenge 1B, this system processes multiple PDF documents simultaneously to deliver personalized, actionable insights.

## Project Structure
```
Challenge_1b/
├── Collection 1/                    # Travel Planning
│   ├── PDFs/                       # South of France guides
│   ├── challenge1b_input.json      # Input configuration
│   └── challenge1b_output.json     # Analysis results
├── Collection 2/                    # Adobe Acrobat Learning
│   ├── PDFs/                       # Acrobat tutorials
│   ├── challenge1b_input.json      # Input configuration
│   └── challenge1b_output.json     # Analysis results
├── Collection 3/                    # Recipe Collection
│   ├── PDFs/                       # Cooking guides
│   ├── challenge1b_input.json      # Input configuration
│   └── challenge1b_output.json     # Analysis results
└── README.md
```

## Project Structure on Laptop
```
Adobe
├──1B_allfiles/
 ├── input/                          # Input directory for PDFs
 │   ├── 6874faecd848a_Adobe_India_Hackathon_-_Challenge_Doc.pdf
 │   └── ALocation-SpecificMobileFramework_PublishedBook.pdf
 ├── output/                         # Output directory for analysis results
 │   ├── 6874faecd848a_Adobe_India_Hackathon_-_Challenge_Doc.json
 │   ├── ALocation-SpecificMobileFramework_PublishedBook.json
 │   ├── analysis_results_20250724_010735.json
 │   ├── analysis_results_20250724_011045.json
 │   ├── analysis_results_20250724_011319.json
 │   ├── analysis_results_20250725_124223.json
 │   ├── analysis_results_20250725_124558.json
 │   ├── analysis_results_20250725_124712.json
 │   └── result.json
 ├── doc_intelligent_system_docker.py  # Main processing script (Docker version)
 ├── doc_intelligent_system_windows.py # Alternative Windows version
 ├── Dockerfile                      # Docker container configuration
 ├── dockerfile.py                   # Docker helper script
 ├── README_1b.md                    # This documentation
 ├── requirements1b.txt              # Python dependencies
 └── sample1.py                      # Sample/test script
```
## Solution Approach

- doc_intelligent_system_docker.py: Primary Docker-compatible version
- doc_intelligent_system_windows.py: Windows-specific implementation
- sample1.py: Testing and development script
- dockerfile.py: Docker build automation helper

## Methodology Overview
Our document intelligence system employs a multi-stage approach to extract and prioritize relevant sections from PDF documents based on user persona and job-to-be-done requirements.

### Core Components

1. Document Processing Pipeline
- PDF Text Extraction: Uses PyPDF2 to extract text page-by-page from input documents
- Section Segmentation: Employs regex patterns to identify logical document sections (Abstract, Introduction, Methodology, etc.)
- Content Filtering: Removes sections with insufficient content (< 50 characters) to focus on substantial material

2. Relevance Scoring Algorithm
The system calculates relevance scores using a hybrid approach:
- Keyword Matching (40% weight): Extracts keywords from persona and job descriptions, then counts occurrences in document sections. This captures direct topical alignment.
- TF-IDF Similarity (60% weight): Implements cosine similarity between the persona-job query and section content using TF-IDF vectorization. This captures semantic relevance beyond exact keyword matches. The weighted combination ensures both literal and conceptual relevance are considered.

3. Content Refinement
- Sentence Scoring: Ranks sentences by position weight (earlier = more important) and keyword density
- Intelligent Summarization: Selects top-scoring sentences within length constraints (500 characters)
- Context Preservation: Maintains coherent narrative flow in refined sections

## Technical Implementation

### Libraries Used

- PyPDF2 3.0.1: Core PDF text extraction engine
- NLTK 3.8.1: Natural language processing toolkit for text tokenization and stopwords
- scikit-learn 1.3.0: TF-IDF vectorization and cosine similarity calculations
- NumPy 1.24.3: Numerical computations and array operations

## Technical Optimizations

### Performance Constraints
- CPU-Only Architecture: Uses lightweight scikit-learn TF-IDF instead of heavy transformer models
- Memory Efficiency: Processes documents sequentially and limits feature vectors to 1000 dimensions
- Speed Optimization: Implements early stopping in text refinement and limits section analysis to top 10 results

### Model Selection Rationale
We chose classical NLP techniques over modern language models due to:
- Model size constraint (≤ 1GB)
- CPU-only requirement
- 60-second processing time limit
- No internet access restriction

## Implementation

### Prerequisites

1. Docker installed on your system
2. PDF documents to analyze
3. Windows PowerShell (for the provided commands)

### Step 1: Navigate to Project Directory
```
powershell cd C:\Users\KIIT\OneDrive\Desktop\Adobe\1B_allfiles
dir
```
### Step 2: Create Requirements File
```
powershell @"
PyPDF2==3.0.1
nltk==3.8.1
scikit-learn==1.3.0
numpy==1.24.3
"@ | Out-File -FilePath "requirements1b.txt" -Encoding UTF8
```
Step 3: Create Dockerfile
```
powershell@"
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements1b.txt .
RUN pip install --no-cache-dir -r requirements1b.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)"

# Copy the main Python script
COPY doc_intelligent_system_docker.py main.py

# Create input and output directories
RUN mkdir -p /app/input /app/output

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PERSONA="Researcher"
ENV JOB="Analyze documents"

# Run the application
CMD ["python", "main.py"]
"@ | Out-File -FilePath "Dockerfile" -Encoding UTF8
```
Step 4: Create Input/Output Directories
```
powershellif (-not (Test-Path "input")) {
    New-Item -ItemType Directory -Name "input"
}
if (-not (Test-Path "output")) {
    New-Item -ItemType Directory -Name "output"  
}
```

### Alternative Execution Options
```
Option 1: Docker Execution (Recommended)
powershelldocker build -t doc-intelligence .
docker run -v "${PWD}/input:/app/input" -v "${PWD}/output:/app/output" doc-intelligence

Option 2: Windows Direct Execution
powershellpython doc_intelligent_system_windows.py

Option 3: Local Development/Testing
powershellpython sample1.py  # For testing and development
```
## Docker Configuration

### Dockerfile Breakdown
```
dockerfileFROM python:3.9-slim                 # Lightweight Python base image
WORKDIR /app                                   # Set working directory
RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*  # Build dependencies
COPY requirements1b.txt .                      # Copy dependencies file
RUN pip install --no-cache-dir -r requirements1b.txt  # Install Python packages
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)"  # Download NLTK data
COPY doc_intelligent_system_docker.py main.py  # Copy main processing script
RUN mkdir -p /app/input /app/output            # Create required directories
ENV PYTHONUNBUFFERED=1                         # Enable real-time logging
ENV PERSONA="Researcher"                       # Default persona
ENV JOB="Analyze documents"                    # Default job description
CMD ["python", "main.py"]                      # Set container entry point
```

### Environment Variables

PERSONA: User role for content analysis (default: "Researcher")
JOB: Task description for relevance scoring (default: "Analyze documents")
PYTHONUNBUFFERED: Enables real-time logging output

## Usage Instructions

### Input Requirements

- PDF Documents: Place PDF files in the input/ directory
- Configuration: Optionally provide challenge1b_input.json with specific persona and job requirements
- Document Types: Supports various PDF formats including academic papers, reports, guides, and manuals

Input JSON Format
```
json{
  "challenge_info": {
    "challenge_id": "round_1b_001",
    "test_case_name": "document_analysis"
  },
  "documents": [
    {"filename": "document1.pdf", "title": "Document Title"}
  ],
  "persona": {
    "role": "Travel Planner"
  },
  "job_to_be_done": {
    "task": "Plan a 4-day trip for 10 college friends to South of France"
  }
}
```

## Analysis Results Overview
Your system has successfully processed multiple document collections with timestamped results:
Recent Analysis Runs (July 25, 2025)

analysis_results_20250725_124712.json: Latest comprehensive analysis
analysis_results_20250725_124558.json: Secondary analysis run
analysis_results_20250725_124223.json: Initial analysis attempt

### Document-Specific Results

6874faecd848a_Adobe_India_Hackathon_-_Challenge_Doc.json: Hackathon documentation analysis
ALocation-SpecificMobileFramework_PublishedBook.json: Mobile framework analysis
result.json: Consolidated final results

Each JSON file contains:

- metadata: Processing context and document information
- extracted_sections: Prioritized relevant sections with importance rankings
- subsection_analysis: Refined content summaries with confidence scores
```
json{
  "metadata": {
    "input_documents": ["document1.pdf", "document2.pdf"],
    "persona": "Travel Planner",
    "job_to_be_done": "Plan a 4-day trip for 10 college friends to South of France",
    "processing_time": 15.23,
    "total_sections_analyzed": 45
  },
  "extracted_sections": [
    {
      "document": "travel_guide.pdf",
      "section_title": "Accommodation Options",
      "importance_rank": 1,
      "relevance_score": 0.89,
      "page_number": 12,
      "content_preview": "Budget-friendly hostels and group bookings..."
    }
  ],
  "subsection_analysis": [
    {
      "document": "travel_guide.pdf",
      "refined_text": "For groups of 10+, consider booking entire apartments or hostels with group discounts. Many locations offer 20-30% savings for advance bookings.",
      "page_number": 12,
      "confidence_score": 0.92
    }
  ]
}
```
## Performance Specifications

### Adobe Hackathon Requirements Compliance

✅ Processing Time: ≤ 60 seconds for multi-document collections
✅ Memory Usage: Optimized for 16GB RAM limit
✅ CPU Utilization: Efficient single-threaded processing
✅ Network Isolation: No internet access required (pre-downloaded NLTK data)
✅ Architecture: AMD64 (x86_64) compatible
✅ Model Size: ≤ 1GB constraint (using classical NLP techniques)
✅ Open Source: All dependencies are open source

### Performance Features

- Intelligent Caching: Reuses TF-IDF vectors for similar content analysis
- Memory Management: Sequential document processing prevents memory overflow
- Progress Tracking: Real-time logging of analysis progress
- Error Recovery: Continues processing even if individual documents fail

## Development and Testing

### Local Development
The solution supports local development outside Docker by setting environment variables:
```
bashset PERSONA=Researcher
set JOB=Analyze technical documents
python doc_intelligent_system_docker.py
```
### Testing Strategy

- Single Document: Test with individual PDF files
- Multi-Collection: Test with diverse document types and personas
- Large Documents: Performance testing with 100+ page PDFs
- Edge Cases: Malformed PDFs, unusual layouts, mixed languages

### Debugging
Enable detailed logging by modifying the logging level in the main script:
```
pythonlogging.basicConfig(level=logging.DEBUG)  # More verbose output
```
## Troubleshooting

### Common Issues

- No Documents Found: Ensure PDFs are in the input/ directory
- NLTK Data Missing: NLTK downloads are included in Docker build
- Memory Errors: Reduce document batch size for large collections
- Low Relevance Scores: Adjust persona and job descriptions for better matching

### Optimization Tips

- Use specific, descriptive persona roles
- Include relevant keywords in job descriptions
- Ensure PDFs have extractable text (not scanned images)
- Verify sufficient disk space for output files

## Architecture Details

### Algorithm Flow

- Input Processing: Parse configuration and load PDF documents
- Text Extraction: Extract text content page-by-page
- Section Detection: Identify document structure using pattern matching
- Relevance Scoring: Calculate importance using hybrid TF-IDF + keyword approach
- Content Refinement: Summarize and rank subsections
- Output Generation: Create structured JSON results

### Scalability Considerations

- Document Limits: Optimized for 10-50 documents per collection
- Content Filtering: Removes low-quality sections to focus on valuable content
- Feature Limiting: TF-IDF vectors capped at 1000 features for performance
- Memory Efficiency: Processes documents individually to prevent overflow

## License
Open source solution using freely available libraries and tools as required by Adobe Hackathon guidelines.

Adobe India Hackathon 2025 - Challenge 1B Submission
---
