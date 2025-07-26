# Challenge 1a: PDF Processing Solution

## Overview
This solution extracts structured document outlines from PDF files, identifying titles and hierarchical headings to generate JSON outputs. Built for the Adobe India Hackathon 2025 Challenge 1A, it processes PDFs within Docker containers while meeting strict performance and resource constraints.

### Key Requirements
- **Automatic Processing**: Process all PDFs from `/app/input` directory
- **Output Format**: Generate `filename.json` for each `filename.pdf`
- **Input Directory**: Read-only access only
- **Open Source**: All libraries, models, and tools must be open source
- **Cross-Platform**: Test on both simple and complex PDFs

## Solution Structure in GitHub
```
Challenge_1a/
├── sample_dataset/
│   ├── outputs/         # JSON files provided as outputs.
│   ├── pdfs/            # Input PDF files
│   └── schema/          # Output schema definition
│       └── output_schema.json
├── Dockerfile           # Docker container configuration
├── process_pdfs.py      #  processing script
└── README.md           # This file
```

## Solution Structure on laptop
```
Adobe/
├── 1A_allfiles/
│   ├── output/            # JSON files provided as outputs.
│   ├── input/             # Input PDF files
│  
├── Dockerfile             # Docker container configuration
├── pdf_processor.py       # Processing script - PDF Outline Extractor Docker Version with Multilingual Support (with docker)
├── main.py                # Processing script
├── main2.py               # Multi-threaded PDF Structure Extractor with Enhanced Title Detection and Text Merging
├── main3.py               # PDF Outline Extractor - Local Version with Multilingual Support (without docker - locally)
├── setup.py               # This setup script checks if PyMuPDF is installed, creates input/output directories, and                                             prepares your environment for extracting PDF table of contents/outlines.
├── requirements1a.txt     # Requirements
└── README_1a.md           # This file
```

## Solution Approach 

### Methodology
- Text Block Extraction: Extract all text spans with font metadata (size, style, position, page)
- Intelligent Block Merging: Merge adjacent spans with similar styling to reconstruct complete text lines
- Enhanced Title Detection: Identify document title using font size, position, and centering analysis
- Multi-Level Heading Detection: Classify bold text as headings with automatic level assignment
- Pattern Recognition: Detect academic patterns (course codes, units, chapters) for better classification
- Structured Output: Generate schema-compliant JSON with title and hierarchical outline

### Key Features
- Universal Environment Support: Runs in both Docker and local development environments
- Multi-threaded Processing: Parallel PDF processing for improved performance
- Advanced Text Merging: Handles complex PDF layouts with spanning text elements
- Comprehensive Metadata: Extracts PDF properties and processing statistics
- Robust Error Handling: Continues processing even when individual PDFs fail
- Performance Monitoring: Tracks processing times and resource usage

## Implementation Guidelines

### Performance Considerations
- **Memory Management**: Efficient handling of large PDFs
- **Processing Speed**: Optimize for sub-10-second execution
- **Resource Usage**: Stay within 16GB RAM constraint
- **CPU Utilization**: Efficient use of 8 CPU cores

### Testing Strategy
- **Simple PDFs**: Test with basic PDF documents
- **Complex PDFs**: Test with multi-column layouts, images, tables
- **Large PDFs**: Verify 50-page processing within time limit

## Technical Implementation 

### Libraries Used
- PyMuPDF (fitz) 1.23.26: Core PDF processing engine for text extraction and font analysis

Python Standard Library:

- json: JSON output generation
- os: File system operations
- logging: Comprehensive logging system
- re: Regular expression pattern matching
- typing: Type hints for better code quality
- concurrent.futures: Multi-threading support
- time: Performance measurement

### Architecture Details

- No ML Models: Pure rule-based approach using document structure analysis
- Memory Efficient: Processes PDFs individually to minimize memory footprint
- CPU Optimized: Adaptive thread pooling based on available CPU cores
- Offline Operation: No network dependencies or external API calls

## Testing My Solution

### Prerequisites

1. Docker installed on your system
2. PDF files to process
3. Windows PowerShell (for the provided commands)

### Step 1: Navigate to Project Directory

powershellcd C:\Users\KIIT\OneDrive\Desktop\Adobe\1A_allfiles
dir

### Step 2: Build Docker Image

powershell @"
FROM python:3.9-slim
WORKDIR /app
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*
COPY requirements1a.txt .
RUN pip install --no-cache-dir -r requirements1a.txt
COPY main.py .
RUN mkdir -p /app/input /app/output
CMD ["python", "main.py"]
"@ | Out-File -FilePath Dockerfile -Encoding UTF8 -NoNewline

docker build -t pdf-processor .

### Step 3: Run Processing

powershelldocker run -v "${PWD}\input:/app/input" -v "${PWD}\output:/app/output" pdf-processor
Docker Configuration
Dockerfile Breakdown
dockerfileFROM python:3.9-slim           # Lightweight Python base image
WORKDIR /app                             # Set working directory
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*  # Install build dependencies
COPY requirements1a.txt .                # Copy dependencies file
RUN pip install --no-cache-dir -r requirements1a.txt  # Install Python packages
COPY main.py .                           # Copy main processing script
RUN mkdir -p /app/input /app/output      # Create required directories
CMD ["python", "main.py"]                # Set container entry point


## Volume Mapping

Input Volume: ${PWD}\input:/app/input - Maps local input directory to container
Output Volume: ${PWD}\output:/app/output - Maps local output directory to container

## Usage Instructions

### Input Requirements

- Place PDF files in the input/ directory
- Ensure PDF files are readable and not password-protected
- Support for various PDF formats and layouts

### Output Format

File Naming: filename.pdf → filename.json

JSON Structure:

json{
  "title": "Document Title",
  "outline": [
    {
      "level": "H1",
      "text": "Chapter 1: Introduction", 
      "page": 1
    }
  ],
  "metadata": {
    "author": "Document Author",
    "creation_date": "2024-01-01"
  },
  "statistics": {
    "total_text_blocks": 1250,
    "total_headings": 15,
    "pages": 25,
    "processing_time": 2.34
  }
}

### Performance Features

- Adaptive Threading: Automatically adjusts thread count based on available CPU cores
- Memory Management: Processes PDFs individually to prevent memory overflow
- Progress Monitoring: Real-time logging of processing status and timing
- Error Recovery: Continues batch processing even if individual files fail

## Development and Testing

### Local Development
The solution supports local development outside Docker:
bashpython main.py  # Runs in local environment (OR bashpython main3.py)

### Testing Strategy

- Simple PDFs: Basic documents with clear structure
- Complex Layouts: Multi-column, academic papers, reports
- Large Documents: 50+ page PDFs for performance testing
- Edge Cases: Malformed PDFs, unusual fonts, mixed languages

### Debugging

Enable detailed logging by modifying the logging level in main.py:
pythonlogging.basicConfig(level=logging.DEBUG)  # More verbose output

## Troubleshooting

### Common Issues

- No PDFs Found: Ensure PDFs are in the input/ directory
- Permission Errors: Check file permissions on input/output directories
- Docker Build Fails: Verify Docker is running and has sufficient resources
- Processing Timeout: Large PDFs may need additional processing time

### Support

- Check logs for detailed error messages
- Verify input PDF file integrity
- Ensure sufficient disk space for output files
- Validate Docker container has proper volume mounts

## License
Open source solution using freely available libraries and tools as required by Adobe Hackathon guidelines.

### Validation Checklist
- [✅] All PDFs in input directory are processed
- [✅] JSON output files are generated for each PDF
- [✅] Output format matches required structure
- [✅] **Output conforms to schema** in `sample_dataset/schema/output_schema.json`
- [✅] Processing completes within 10 seconds for 50-page PDFs
- [✅] Solution works without internet access
- [✅] Memory usage stays within 16GB limit
- [✅] Compatible with AMD64 architecture

Adobe India Hackathon 2025 - Challenge 1A Submission

---
