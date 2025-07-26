# Multi-threaded PDF Structure Extractor with Enhanced Title Detection and Text Merging

import fitz  # PyMuPDF
import os
import json
import logging
import re
import time
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
# Extract Text Blocks
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

                        font_flags = span.get("flags", 0)
                        is_bold = bool(font_flags & (1 << 4)) or 'Bold' in span["font"]
                        is_italic = bool(font_flags & (1 << 1)) or 'Italic' in span["font"]

                        text_blocks.append({
                            "text": text,
                            "font_size": round(span["size"], 2),
                            "font": span["font"],
                            "bold": is_bold,
                            "italic": is_italic,
                            "x": round(span["bbox"][0], 2),
                            "y": round(span["bbox"][1], 2),
                            "page": page_num,
                            "page_width": page_width,
                        })

        doc.close()
        return text_blocks
    except Exception as e:
        logger.error(f"❌ Error processing {pdf_path}: {e}")
        return []


# ----------------------------
# Merge Adjacent Span Blocks
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

        current_text_width = len(current['text']) * current['font_size'] * 0.6
        gap = blk['x'] - (current['x'] + current_text_width)
        adjacent_x = 0 <= gap <= x_gap_threshold

        if same_line and same_style and same_font and adjacent_x:
            current['text'] += " " + blk['text']
        else:
            merged.append(current)
            current = blk.copy()

    merged.append(current)
    return merged


# ----------------------------
# Title Detection
# ----------------------------
def detect_title(text_blocks: List[Dict]) -> Tuple[Optional[str], Optional[float]]:
    first_page_blocks = [b for b in text_blocks if b['page'] == 1 and b['font_size'] >= 10]
    if not first_page_blocks:
        return None, None

    max_font_size = max(b['font_size'] for b in first_page_blocks)
    candidate_blocks = [b for b in first_page_blocks if b['font_size'] == max_font_size and b['y'] <= 200]

    if not candidate_blocks:
        return None, None

    centered_candidates = []
    for b in candidate_blocks:
        approx_center_x = b['x'] + (len(b['text']) * b['font_size'] * 0.3)
        page_center = b['page_width'] / 2
        if abs(approx_center_x - page_center) < 100:
            centered_candidates.append(b)

    if not centered_candidates:
        centered_candidates = candidate_blocks

    merged = merge_blocks(centered_candidates)
    if not merged:
        return None, None

    merged.sort(key=lambda b: b['y'])
    title_text = merged[0]['text'].strip()
    return title_text, max_font_size


# ----------------------------
# Heading Detection Utilities
# ----------------------------
def heading_level_from_text(text: str, default_level: str = "H4") -> str:
    cleaned_text = text.strip().lower()

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


def is_likely_heading(text: str) -> bool:
    if not text or len(text.strip()) == 0:
        return False
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

    grouped_by_font = {}
    for b in heading_blocks:
        grouped_by_font.setdefault(b['font_size'], []).append(b)

    sizes_sorted = sorted(grouped_by_font.keys(), reverse=True)
    level_names = ["H1", "H2", "H3", "H4"]
    heading_levels = {
        size: level_names[i] for i, size in enumerate(sizes_sorted[:len(level_names)])
    }

    final_headings = []

    for size in heading_levels:
        group = grouped_by_font[size]
        merged = merge_blocks(group)

        for b in merged:
            if is_likely_heading(b['text']):
                detected_level = heading_level_from_text(b['text'], heading_levels[size])
                final_headings.append({
                    "level": detected_level,
                    "text": b['text'].strip(),
                    "page": b['page'],
                    "y": b['y']
                })

    final_headings.sort(key=lambda h: (h['page'], h['y']))
    return [{"level": h["level"], "text": h["text"], "page": h["page"]} for h in final_headings]


# ----------------------------
# Single PDF Processor
# ----------------------------
def process_pdf(pdf_path: str) -> Dict:
    logger.info(f"📄 Processing: {os.path.basename(pdf_path)}")
    start_time = time.time()

    blocks = extract_text_blocks(pdf_path)
    if not blocks:
        return {"title": "", "outline": []}

    title, title_font_size = detect_title(blocks)
    headings = detect_headings(blocks, title_font_size)

    elapsed = time.time() - start_time
    logger.info(f"⏱️ Processed {os.path.basename(pdf_path)} in {elapsed:.2f} seconds")

    return {
        "title": title or "",
        "outline": headings
    }


# ----------------------------
# Batch Processor with Threads
# ----------------------------
def process_pdfs(input_dir: str, output_dir: str, max_workers: int = 4) -> None:
    start_batch = time.time()
    os.makedirs(output_dir, exist_ok=True)

    pdfs = [f for f in os.listdir(input_dir) if f.lower().endswith(".pdf")]
    if not pdfs:
        logger.warning("⚠️ No PDFs found in ./input")
        return

    def worker(file: str) -> Tuple[str, bool]:
        try:
            input_path = os.path.join(input_dir, file)
            output_path = os.path.join(output_dir, os.path.splitext(file)[0] + ".json")

            result = process_pdf(input_path)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4, ensure_ascii=False)

            logger.info(f"✅ Output: {output_path}")
            return (file, True)
        except Exception as e:
            logger.error(f"❌ Failed on {file}: {e}")
            return (file, False)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker, file) for file in pdfs]
        for future in as_completed(futures):
            future.result()

    total_batch_time = time.time() - start_batch
    logger.info(f"🎉 Completed processing all PDFs in {total_batch_time:.2f} seconds")


# ----------------------------
# Main Entrypoint with Total Timer
# ----------------------------
def main():
    start = time.time()  # Start total execution timer

    base = os.getcwd()
    input_dir = os.path.join(base, "input")
    output_dir = os.path.join(base, "output")

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    max_threads = 4  # Adjust based on your CPU
    process_pdfs(input_dir, output_dir, max_workers=max_threads)

    end = time.time()
    total_time = end - start
    logger.info(f"✅ Total execution time: {total_time:.2f} seconds")
    logger.info("🎉 Extraction complete.")


if __name__ == "__main__":
    main()
