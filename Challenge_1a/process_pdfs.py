# PDF Outline Extractor - Docker Version with Multilingual Support (with docker)

# saved as pdf_processor.py

import fitz  # PyMuPDF
import os
import json
import logging
import re
import time
import unicodedata
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# ----------------------------
# Logging Setup
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ----------------------------
# Language Detection Utilities
# ----------------------------
def detect_language(text: str) -> str:
    """
    Simple language detection based on character sets.
    Returns 'ja' for Japanese, 'en' for English.
    """
    if not text:
        return 'en'
    
    # Count Japanese characters
    japanese_chars = 0
    total_chars = 0
    
    for char in text:
        if char.isspace():
            continue
        total_chars += 1
        
        # Check for Hiragana, Katakana, and Kanji
        if (0x3040 <= ord(char) <= 0x309F or  # Hiragana
            0x30A0 <= ord(char) <= 0x30FF or  # Katakana
            0x4E00 <= ord(char) <= 0x9FFF or  # Kanji (CJK Unified Ideographs)
            0x3400 <= ord(char) <= 0x4DBF):   # Kanji Extension A
            japanese_chars += 1
    
    if total_chars == 0:
        return 'en'
    
    japanese_ratio = japanese_chars / total_chars
    return 'ja' if japanese_ratio > 0.3 else 'en'


def normalize_japanese_text(text: str) -> str:
    """
    Normalize Japanese text for better processing.
    """
    # Normalize Unicode (NFC normalization)
    text = unicodedata.normalize('NFC', text)
    
    # Convert full-width numbers and letters to half-width
    text = text.translate(str.maketrans(
        '０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ',
        '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    ))
    
    # Convert full-width spaces to half-width
    text = text.replace('　', ' ')
    
    return text.strip()


# ----------------------------
# Extract Text Blocks (Enhanced for Japanese)
# ----------------------------
def extract_text_blocks(pdf_path: str) -> List[Dict]:
    try:
        doc = fitz.open(pdf_path)
        text_blocks = []

        for page_num, page in enumerate(doc, start=1):
            page_width = page.rect.width
            page_dict = page.get_text("dict")
            blocks = page_dict["blocks"]

            for block in blocks:
                if "lines" not in block:
                    continue

                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if not text:
                            continue

                        # Normalize Japanese text
                        normalized_text = normalize_japanese_text(text)
                        if not normalized_text:
                            continue

                        font_flags = span.get("flags", 0)
                        is_bold = bool(font_flags & (1 << 4)) or 'Bold' in span["font"]
                        is_italic = bool(font_flags & (1 << 1)) or 'Italic' in span["font"]
                        
                        # Detect language
                        language = detect_language(normalized_text)

                        text_blocks.append({
                            "text": normalized_text,
                            "original_text": text,
                            "font_size": round(span["size"], 2),
                            "font": span["font"],
                            "bold": is_bold,
                            "italic": is_italic,
                            "x": round(span["bbox"][0], 2),
                            "y": round(span["bbox"][1], 2),
                            "page": page_num,
                            "page_width": page_width,
                            "language": language,
                        })

        doc.close()
        return text_blocks
    except Exception as e:
        logger.error(f"❌ Error processing {pdf_path}: {e}")
        return []


# ----------------------------
# Enhanced Merge Function for Japanese
# ----------------------------
def merge_blocks(blocks, y_tolerance=3, x_gap_threshold=10):
    if len(blocks) == 0:
        return []

    blocks = sorted(blocks, key=lambda b: (round(b['y']), b['x']))
    merged = []
    current = blocks[0].copy()

    for blk in blocks[1:]:
        same_line = abs(blk['y'] - current['y']) <= y_tolerance
        same_style = blk['font_size'] == current['font_size'] and blk['bold'] == current['bold']
        same_font = blk['font'] == current['font']
        same_language = blk['language'] == current['language']

        # Adjust gap calculation for Japanese (characters can be closer)
        if current['language'] == 'ja':
            current_text_width = len(current['text']) * current['font_size'] * 0.8  # Japanese chars are wider
            gap_threshold = x_gap_threshold * 1.5  # Allow larger gaps for Japanese
        else:
            current_text_width = len(current['text']) * current['font_size'] * 0.6
            gap_threshold = x_gap_threshold

        gap = blk['x'] - (current['x'] + current_text_width)
        adjacent_x = 0 <= gap <= gap_threshold

        if same_line and same_style and same_font and same_language and adjacent_x:
            # Add appropriate spacing for Japanese
            if current['language'] == 'ja':
                # Japanese doesn't always need spaces between merged text
                current['text'] += blk['text']
            else:
                current['text'] += " " + blk['text']
        else:
            merged.append(current)
            current = blk.copy()

    merged.append(current)
    return merged


# ----------------------------
# Enhanced Title Detection
# ----------------------------
def detect_title(text_blocks: List[Dict]) -> Tuple[Optional[str], Optional[float], Optional[str]]:
    first_page_blocks = [b for b in text_blocks if b['page'] == 1 and b['font_size'] >= 10]
    if not first_page_blocks:
        return None, None, None

    max_font_size = max(b['font_size'] for b in first_page_blocks)
    candidate_blocks = [b for b in first_page_blocks if b['font_size'] == max_font_size and b['y'] <= 200]

    if not candidate_blocks:
        return None, None, None

    # Find centered candidates with language-aware centering
    centered_candidates = []
    for b in candidate_blocks:
        if b['language'] == 'ja':
            # Japanese text centering calculation
            approx_center_x = b['x'] + (len(b['text']) * b['font_size'] * 0.4)
        else:
            approx_center_x = b['x'] + (len(b['text']) * b['font_size'] * 0.3)
        
        page_center = b['page_width'] / 2
        if abs(approx_center_x - page_center) < 100:
            centered_candidates.append(b)

    if not centered_candidates:
        centered_candidates = candidate_blocks

    merged = merge_blocks(centered_candidates)
    if not merged:
        return None, None, None

    merged.sort(key=lambda b: b['y'])
    title_text = merged[0]['text'].strip()
    title_language = merged[0]['language']
    return title_text, max_font_size, title_language


# ----------------------------
# Multilingual Heading Detection
# ----------------------------
def heading_level_from_text(text: str, language: str, default_level: str = "H4") -> str:
    """
    Detect heading level from text content with language-specific patterns.
    """
    cleaned_text = text.strip().lower()
    
    if language == 'ja':
        # Japanese patterns
        # Chapter patterns: 第1章, 第一章, 1章, etc.
        if re.search(r'第[一二三四五六七八九十\d]+章', text):
            return "H1"
        if re.search(r'[一二三四五六七八九十\d]+章', text):
            return "H1"
        
        # Section patterns: 第1節, 1節, etc.
        if re.search(r'第[一二三四五六七八九十\d]+節', text):
            return "H2"
        if re.search(r'[一二三四五六七八九十\d]+節', text):
            return "H2"
        
        # Japanese unit patterns
        if re.search(r'第[一二三四五六七八九十\d]+単元', text):
            return "H2"
        if re.search(r'[一二三四五六七八九十\d]+単元', text):
            return "H2"
        
        # Common Japanese headings
        if any(word in text for word in ['目的', '目標', 'ねらい', '学習目標']):
            return "H2"
        if any(word in text for word in ['概要', 'まとめ', '要約', 'おわりに']):
            return "H2"
        if any(word in text for word in ['はじめに', '序論', '導入']):
            return "H2"
        
        # Numbered patterns: 1. 2. 3. or １．２．３．
        if re.match(r'^[1-9一二三四五六七八九十１２３４５６７８９０][\.\．]', text):
            return "H3"
            
    else:
        # English patterns (original logic)
        if re.match(r"^\(r\d{2}[a-z]+\d{4}\)\s+.+$", text, re.IGNORECASE):
            return "H1"
        if "objectives" in cleaned_text:
            return "H2"
        if "outcomes" in cleaned_text:
            return "H2"
        if re.match(r"^unit[\s\-]*[iivxcl\d]+$", cleaned_text, re.IGNORECASE):
            return "H3"
        if re.match(r"^(chapter|module)[\s\-]*[iivxcl\d]+$", cleaned_text, re.IGNORECASE):
            return "H3"

    return default_level


def is_likely_heading(text: str, language: str) -> bool:
    """
    Determine if text is likely a heading with language-specific rules.
    """
    if not text or len(text.strip()) == 0:
        return False
    
    if language == 'ja':
        # Japanese heading criteria
        # Length check (Japanese headings can be longer due to character density)
        if len(text) > 150:
            return False
        
        # Japanese headings rarely end with these characters
        if text.endswith(('。', '、', '？')):
            return False
            
        # Check for typical Japanese heading patterns
        has_chapter_pattern = bool(re.search(r'第?[一二三四五六七八九十\d]+[章節単元]', text))
        has_numbered_pattern = bool(re.match(r'^[1-9一二三四五六七八九十１２３４５６７８９０][\.\．]', text))
        has_heading_words = any(word in text for word in [
            '目的', '目標', 'ねらい', '概要', 'まとめ', '要約', 
            'はじめに', '序論', '導入', 'おわりに', '結論'
        ])
        
        # More lenient for Japanese - check character density and patterns
        if has_chapter_pattern or has_numbered_pattern or has_heading_words:
            return True
            
        # For other Japanese text, be more restrictive about length
        if len(text) > 50:
            return False
            
    else:
        # English criteria (original logic)
        if len(text) > 100 or text.count(" ") > 15:
            return False
        if text.endswith(".") or text.endswith("?"):
            return False
        if text[0].islower():
            return False
    
    return True


def detect_headings(text_blocks: List[Dict], title_font_size: Optional[float]) -> List[Dict]:
    excluded_size = {title_font_size} if title_font_size else set()
    min_heading_size = 12

    heading_blocks = [
        b for b in text_blocks
        if b['font_size'] not in excluded_size and
           b['font_size'] >= min_heading_size and
           b['bold'] and
           b['y'] < 700
    ]

    # Group by both font size and language
    grouped_by_font_lang = {}
    for b in heading_blocks:
        key = (b['font_size'], b['language'])
        grouped_by_font_lang.setdefault(key, []).append(b)

    # Sort sizes within each language group
    sizes_sorted = sorted(set(fs for fs, lang in grouped_by_font_lang.keys()), reverse=True)
    level_names = ["H1", "H2", "H3", "H4"]
    
    final_headings = []

    for (font_size, lang) in grouped_by_font_lang:
        size_rank = sizes_sorted.index(font_size)
        if size_rank < len(level_names):
            default_level = level_names[size_rank]
        else:
            default_level = "H4"

        group = grouped_by_font_lang[(font_size, lang)]
        merged = merge_blocks(group)

        for b in merged:
            if is_likely_heading(b['text'], b['language']):
                detected_level = heading_level_from_text(b['text'], b['language'], default_level)
                final_headings.append({
                    "level": detected_level,
                    "text": b['text'].strip(),
                    "page": b['page'],
                    "y": b['y'],
                    "language": b['language']
                })

    final_headings.sort(key=lambda h: (h['page'], h['y']))
    return [{"level": h["level"], "text": h["text"], "page": h["page"], "language": h["language"]} for h in final_headings]


# ----------------------------
# Enhanced PDF Processor
# ----------------------------
def process_pdf(pdf_path: str) -> Dict:
    logger.info(f"📄 Processing: {os.path.basename(pdf_path)}")
    start_time = time.time()

    blocks = extract_text_blocks(pdf_path)
    if not blocks:
        return {"title": "", "title_language": "en", "outline": [], "document_language": "en"}

    title, title_font_size, title_language = detect_title(blocks)
    headings = detect_headings(blocks, title_font_size)
    
    # Determine dominant document language
    languages = [b['language'] for b in blocks]
    document_language = max(set(languages), key=languages.count) if languages else 'en'

    elapsed = time.time() - start_time
    logger.info(f"⏱️ Processed {os.path.basename(pdf_path)} in {elapsed:.2f} seconds")

    return {
        "title": title or "",
        "title_language": title_language or document_language,
        "outline": headings,
        "document_language": document_language
    }


# ----------------------------
# Batch Processor (Docker-optimized)
# ----------------------------
def process_pdfs(input_dir: str, output_dir: str, max_workers: int = 4) -> None:
    start_batch = time.time()
    os.makedirs(output_dir, exist_ok=True)

    # Ensure directories exist
    if not os.path.exists(input_dir):
        logger.error(f"❌ Input directory does not exist: {input_dir}")
        return

    pdfs = [f for f in os.listdir(input_dir) if f.lower().endswith(".pdf")]
    if not pdfs:
        logger.warning(f"⚠️ No PDFs found in {input_dir}")
        return

    logger.info(f"🔍 Found {len(pdfs)} PDF(s) to process")

    def worker(file: str) -> Tuple[str, bool]:
        try:
            input_path = os.path.join(input_dir, file)
            output_path = os.path.join(output_dir, os.path.splitext(file)[0] + ".json")

            result = process_pdf(input_path)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4, ensure_ascii=False)

            logger.info(f"✅ Output: {os.path.basename(output_path)}")
            return (file, True)
        except Exception as e:
            logger.error(f"❌ Failed on {file}: {e}")
            return (file, False)

    successful = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker, file) for file in pdfs]
        for future in as_completed(futures):
            _, success = future.result()
            if success:
                successful += 1
            else:
                failed += 1

    total_batch_time = time.time() - start_batch
    logger.info(f"🎉 Completed processing: {successful} successful, {failed} failed")
    logger.info(f"⏱️ Total batch time: {total_batch_time:.2f} seconds")


# ----------------------------
# Main Entrypoint (Docker-compatible)
# ----------------------------
def main():
    start = time.time()
    
    # Docker paths
    input_dir = "/app/input"
    output_dir = "/app/output"
    
    # Ensure directories exist
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("🚀 Starting PDF processing in Docker container")
    logger.info(f"📁 Input directory: {input_dir}")
    logger.info(f"📁 Output directory: {output_dir}")

    max_threads = 4
    process_pdfs(input_dir, output_dir, max_workers=max_threads)

    end = time.time()
    total_time = end - start
    logger.info(f"✅ Total execution time: {total_time:.2f} seconds")
    logger.info("🎉 PDF extraction complete.")


if __name__ == "__main__":
    main()
