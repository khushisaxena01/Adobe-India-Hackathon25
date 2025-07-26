# Comprehensive PDF Metadata and Structure Analyzer with Statistical Reporting

import fitz  # PyMuPDF
import os
import json
import logging
from typing import List, Dict, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_text_blocks(pdf_path: str) -> List[Dict]:
    """
    Extract text blocks with metadata: font size, font name, bold, text, position, page number.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of dictionaries containing text block information
    """
    try:
        doc = fitz.open(pdf_path)
        text_blocks = []

        for page_num, page in enumerate(doc, start=1):
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" not in block:  # Skip image blocks
                    continue
                    
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span["text"].strip()
                        if not text:
                            continue
                            
                        # Enhanced font detection
                        font_flags = span.get("flags", 0)
                        is_bold = bool(font_flags & 2**4) or 'Bold' in span["font"]
                        is_italic = bool(font_flags & 2**1) or 'Italic' in span["font"]
                        
                        block_data = {
                            "text": text,
                            "font_size": round(span["size"], 2),
                            "font": span["font"],
                            "bold": is_bold,
                            "italic": is_italic,
                            "x": round(span["bbox"][0], 2),
                            "y": round(span["bbox"][1], 2),
                            "width": round(span["bbox"][2] - span["bbox"][0], 2),
                            "height": round(span["bbox"][3] - span["bbox"][1], 2),
                            "page": page_num
                        }
                        text_blocks.append(block_data)
        
        doc.close()
        return text_blocks
        
    except Exception as e:
        logger.error(f"Error extracting text blocks from {pdf_path}: {str(e)}")
        return []


def detect_title(text_blocks: List[Dict]) -> Tuple[Optional[str], Optional[float]]:
    """
    Detect the title based on largest font size on the first page, bold preference, and top position.
    
    Args:
        text_blocks: List of text block dictionaries
        
    Returns:
        Tuple of (title_text, title_font_size)
    """
    first_page_blocks = [b for b in text_blocks if b['page'] == 1]
    if not first_page_blocks:
        return None, None

    # Filter out very small text (likely headers/footers)
    min_font_size = 10
    first_page_blocks = [b for b in first_page_blocks if b['font_size'] >= min_font_size]
    
    if not first_page_blocks:
        return None, None

    max_font_size = max(b['font_size'] for b in first_page_blocks)
    candidates = [b for b in first_page_blocks if b['font_size'] == max_font_size]

    # Prefer bold, centered, and top of the page (small y)
    def title_score(block):
        score = 0
        if block['bold']:
            score += 10
        # Prefer text that's more centered (assuming page width ~600)
        if 200 <= block['x'] <= 400:
            score += 5
        # Prefer text in upper portion of page
        if block['y'] <= 200:
            score += 3
        return score

    candidates.sort(key=lambda b: (-title_score(b), b['y']))
    
    title_block = candidates[0]
    return title_block['text'], title_block['font_size']


def detect_headings(text_blocks: List[Dict], title_font_size: Optional[float]) -> List[Dict]:
    """
    Detect headings (H1, H2, H3) based on font sizes, skipping the title font size.
    
    Args:
        text_blocks: List of text block dictionaries
        title_font_size: Font size of the detected title
        
    Returns:
        List of heading dictionaries
    """
    # Collect font sizes excluding title font size and very small fonts
    min_heading_size = 11  # Minimum size for headings
    excluded_sizes = {title_font_size} if title_font_size else set()
    
    sizes = sorted({
        b['font_size'] for b in text_blocks 
        if b['font_size'] not in excluded_sizes 
        and b['font_size'] >= min_heading_size
        and b['bold']  # Only consider bold text as potential headings
    }, reverse=True)

    # Map sizes to heading levels
    heading_levels = {}
    level_names = ["H1", "H2", "H3", "H4", "H5", "H6"]
    for i, size in enumerate(sizes[:len(level_names)]):
        heading_levels[size] = level_names[i]

    headings = []
    for block in text_blocks:
        size = block['font_size']
        if size in heading_levels:
            # Additional validation for headings
            text = block['text']
            # Skip very long text (likely not headings)
            if len(text) > 100:
                continue
            # Skip text that looks like running text
            if text.endswith('.') and len(text) > 50:
                continue
                
            headings.append({
                "level": heading_levels[size],
                "text": text,
                "page": block['page'],
                "font_size": size,
                "position": {"x": block['x'], "y": block['y']}
            })

    return headings


def extract_metadata(pdf_path: str) -> Dict:
    """
    Extract PDF metadata.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Dictionary containing PDF metadata
    """
    try:
        doc = fitz.open(pdf_path)
        metadata = doc.metadata
        doc.close()
        
        return {
            "title": metadata.get("title", ""),
            "author": metadata.get("author", ""),
            "subject": metadata.get("subject", ""),
            "creator": metadata.get("creator", ""),
            "producer": metadata.get("producer", ""),
            "creation_date": metadata.get("creationDate", ""),
            "modification_date": metadata.get("modDate", "")
        }
    except Exception as e:
        logger.error(f"Error extracting metadata from {pdf_path}: {str(e)}")
        return {}


def process_pdf(pdf_path: str) -> Dict:
    """
    Process a single PDF file and extract its structure.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Dictionary containing extracted structure
    """
    logger.info(f"Processing {os.path.basename(pdf_path)}")
    
    text_blocks = extract_text_blocks(pdf_path)
    if not text_blocks:
        logger.warning(f"No text blocks found in {pdf_path}")
        return {"title": "", "outline": [], "metadata": {}}
    
    title, title_font_size = detect_title(text_blocks)
    headings = detect_headings(text_blocks, title_font_size)
    metadata = extract_metadata(pdf_path)
    
    return {
        "title": title if title else "",
        "outline": headings,
        "metadata": metadata,
        "statistics": {
            "total_text_blocks": len(text_blocks),
            "total_headings": len(headings),
            "pages": max(b['page'] for b in text_blocks) if text_blocks else 0
        }
    }


def process_pdfs(input_dir: str, output_dir: str) -> None:
    """
    Process all PDF files in the input directory.
    
    Args:
        input_dir: Directory containing PDF files
        output_dir: Directory to save JSON output files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".pdf")]
    
    if not pdf_files:
        logger.warning(f"No PDF files found in {input_dir}")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    for filename in pdf_files:
        pdf_path = os.path.join(input_dir, filename)
        
        try:
            result = process_pdf(pdf_path)
            
            json_filename = os.path.splitext(filename)[0] + ".json"
            output_path = os.path.join(output_dir, json_filename)
            
            with open(output_path, "w", encoding='utf-8') as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
            
            logger.info(f"✓ Processed {filename} --> {json_filename}")
            
        except Exception as e:
            logger.error(f"✗ Failed to process {filename}: {str(e)}")


def main():
    """Main function to run the PDF structure extractor."""
    BASE_DIR = os.getcwd()
    INPUT_DIR = os.path.join(BASE_DIR, "input")
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")
    
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)
        logger.info(f"Created input directory: {INPUT_DIR}")
        logger.info("Please add PDFs to the input directory and rerun the script.")
        return
    
    process_pdfs(INPUT_DIR, OUTPUT_DIR)
    logger.info(f"Processing complete. JSON files saved in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()