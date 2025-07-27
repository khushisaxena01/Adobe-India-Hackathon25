# PDF Structure Extractor - Round 1A

## Approach

This solution extracts structured outlines from PDF documents using PyMuPDF (fitz) to identify titles and hierarchical headings (H1, H2, H3).

### Methodology

1. **Text Block Extraction**: Extract all text spans with font properties (size, style, position)
2. **Block Merging**: Merge adjacent spans with similar styling to form complete text lines
3. **Title Detection**: Identify the largest font text on the first page within the header area
4. **Heading Detection**: Classify bold text with font sizes >= 12pt as headings, excluding title font size
5. **Level Assignment**: Map font sizes to heading levels (H1=largest, H2=second largest, etc.)
6. **Output Generation**: Create JSON structure with title and ordered heading outline

### Key Features

- Font size and style-based heading detection
- Robust text merging to handle multi-span headings  
- Position-based filtering to avoid footers/noise
- Multilingual text support via UTF-8 encoding
- Heuristic filtering for heading-like text patterns

## Libraries Used

- **PyMuPDF (fitz)**: PDF text extraction and font analysis
- **Standard Python**: json, os, logging, re, typing

## Model Information

No ML models used - purely rule-based approach using font properties and document structure analysis.

## Build and Run Instructions

### Build Docker Image
```bash
docker build --platform linux/amd64 -t pdf-extractor:v1 .
```

### Run Solution
```bash
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none pdf-extractor:v1
```

### Input/Output
- **Input**: Place PDF files in `./input/` directory
- **Output**: JSON files generated in `./output/` directory (filename.pdf → filename.json)

## Architecture Compatibility

- Built for AMD64 (x86_64) architecture
- CPU-only processing, no GPU dependencies
- Offline operation with no network calls
- Optimized for 8 CPU, 16GB RAM systems