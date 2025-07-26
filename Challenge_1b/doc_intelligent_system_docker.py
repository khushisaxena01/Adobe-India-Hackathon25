#!/usr/bin/env python3
"""
Document Intelligence System for Adobe Hackathon 1B - Docker Version
Optimized for containerized deployment with pre-downloaded dependencies
"""

import json
import os
import re
import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple
import argparse
import sys

# Configure logging for Docker environment
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Import required libraries
import PyPDF2
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class DocumentIntelligenceSystem:
    def __init__(self):
        """Initialize with pre-configured NLTK data for Docker"""
        logger.info("Initializing Document Intelligence System...")
        
        # NLTK data should already be downloaded in Docker image
        try:
            self.stop_words = set(stopwords.words('english'))
            logger.info("NLTK resources loaded successfully")
        except LookupError:
            logger.warning("NLTK data not found, downloading...")
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt', quiet=True)
            self.stop_words = set(stopwords.words('english'))
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            max_df=0.95,
            min_df=2
        )
        logger.info("System initialized successfully")
    
    def extract_text_from_pdf(self, pdf_path: str) -> Dict[int, str]:
        """Extract text from PDF with enhanced error handling"""
        page_texts = {}
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                logger.info(f"Extracting text from {total_pages} pages")
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        text = page.extract_text()
                        if text and text.strip():
                            # Clean up text
                            text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
                            page_texts[page_num] = text
                    except Exception as e:
                        logger.warning(f"Error extracting page {page_num}: {e}")
                        continue
                        
                logger.info(f"Successfully extracted text from {len(page_texts)} pages")
                        
        except Exception as e:
            logger.error(f"Error reading PDF {pdf_path}: {e}")
            
        return page_texts
    
    def segment_into_sections(self, text: str, page_num: int) -> List[Dict[str, Any]]:
        """Advanced section segmentation with multiple strategies"""
        sections = []
        
        # Enhanced section patterns for academic and business documents
        section_patterns = [
            # Academic patterns
            r'\n\s*(?:Abstract|ABSTRACT)\s*\n',
            r'\n\s*(?:Introduction|INTRODUCTION)\s*\n',
            r'\n\s*(?:Literature Review|LITERATURE REVIEW)\s*\n',
            r'\n\s*(?:Methodology|METHODOLOGY|Methods|METHODS)\s*\n',
            r'\n\s*(?:Results|RESULTS|Findings|FINDINGS)\s*\n',
            r'\n\s*(?:Discussion|DISCUSSION|Analysis|ANALYSIS)\s*\n',
            r'\n\s*(?:Conclusion|CONCLUSION|Conclusions|CONCLUSIONS)\s*\n',
            r'\n\s*(?:References|REFERENCES|Bibliography|BIBLIOGRAPHY)\s*\n',
            
            # Business patterns
            r'\n\s*(?:Executive Summary|EXECUTIVE SUMMARY)\s*\n',
            r'\n\s*(?:Financial Highlights|FINANCIAL HIGHLIGHTS)\s*\n',
            r'\n\s*(?:Revenue|REVENUE)\s*\n',
            r'\n\s*(?:Market Analysis|MARKET ANALYSIS)\s*\n',
            r'\n\s*(?:Risk Factors|RISK FACTORS)\s*\n',
            
            # General patterns
            r'\n\s*\d+\.\s+[A-Z][^.\n]{10,}\n',  # Numbered sections
            r'\n\s*[A-Z][A-Z\s]{5,}\n',          # ALL CAPS headers
            r'\n\s*[A-Z][^.\n]{15,}\n'           # Title case headers
        ]
        
        # Find all section breaks
        section_breaks = []
        for pattern in section_patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE))
            for match in matches:
                title = match.group().strip()
                if len(title) > 0:
                    section_breaks.append({
                        'position': match.start(),
                        'title': title,
                        'end_position': match.end()
                    })
        
        # Sort by position
        section_breaks.sort(key=lambda x: x['position'])
        
        # Create sections
        if not section_breaks:
            # Fallback: split by paragraph breaks
            paragraphs = text.split('\n\n')
            chunk_size = max(1, len(paragraphs) // 5)  # Aim for ~5 sections
            for i in range(0, len(paragraphs), chunk_size):
                chunk = '\n\n'.join(paragraphs[i:i+chunk_size])
                if len(chunk.strip()) > 100:
                    sections.append({
                        'title': f"Section {len(sections) + 1}",
                        'text': chunk.strip(),
                        'page_number': page_num,
                        'word_count': len(chunk.split())
                    })
        else:
            # Create sections from detected breaks
            for i in range(len(section_breaks)):
                start_pos = section_breaks[i]['end_position']
                end_pos = section_breaks[i + 1]['position'] if i + 1 < len(section_breaks) else len(text)
                
                section_text = text[start_pos:end_pos].strip()
                
                if len(section_text) > 100:  # Minimum section length
                    sections.append({
                        'title': section_breaks[i]['title'],
                        'text': section_text,
                        'page_number': page_num,
                        'word_count': len(section_text.split())
                    })
        
        return sections
    
    def calculate_relevance_score(self, section_text: str, persona: str, job: str) -> float:
        """Enhanced relevance scoring with multiple factors"""
        
        # Create comprehensive query
        query = f"{persona} {job}".lower()
        
        # Extract keywords with better filtering
        persona_keywords = self.extract_keywords(persona.lower())
        job_keywords = self.extract_keywords(job.lower())
        all_keywords = persona_keywords + job_keywords
        
        section_lower = section_text.lower()
        
        # 1. Keyword matching score (weighted by keyword importance)
        keyword_score = 0
        total_keywords = len(all_keywords)
        
        if total_keywords > 0:
            for keyword in all_keywords:
                # Count occurrences and weight by keyword length
                occurrences = section_lower.count(keyword)
                weight = min(len(keyword) / 10.0, 1.0)  # Longer keywords are more important
                keyword_score += occurrences * weight
            
            keyword_score = keyword_score / total_keywords
        
        # 2. TF-IDF similarity score
        similarity_score = 0
        try:
            if len(section_text) > 20:  # Minimum text length for TF-IDF
                corpus = [query, section_text.lower()]
                tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
                similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except Exception as e:
            logger.debug(f"TF-IDF calculation failed: {e}")
            similarity_score = 0
        
        # 3. Section length bonus (moderate length preferred)
        word_count = len(section_text.split())
        length_score = min(1.0, word_count / 500.0) * (1.0 - min(1.0, word_count / 2000.0))
        
        # Combine scores with weights
        final_score = (
            keyword_score * 0.4 +      # Direct keyword matching
            similarity_score * 0.5 +   # Semantic similarity
            length_score * 0.1         # Section length optimization
        )
        
        return final_score
    
    def extract_keywords(self, text: str) -> List[str]:
        """Enhanced keyword extraction"""
        try:
            words = word_tokenize(text.lower())
            # Filter out stopwords, short words, and non-alphabetic tokens
            keywords = [
                word for word in words 
                if (word.isalpha() and 
                    len(word) > 3 and 
                    word not in self.stop_words and
                    not word.isdigit())
            ]
            return list(set(keywords))  # Remove duplicates
        except Exception:
            # Fallback tokenization
            words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
            return list(set(words))
    
    def refine_section_text(self, text: str, max_length: int = 500) -> str:
        """Advanced text refinement with sentence ranking"""
        try:
            sentences = sent_tokenize(text)
        except Exception:
            # Fallback sentence splitting
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(text) <= max_length:
            return text.strip()
        
        # Score sentences using multiple factors
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            if len(sentence.strip()) < 10:  # Skip very short sentences
                continue
                
            # Position score (first and last sentences are important)
            if i == 0:
                position_score = 1.0
            elif i == len(sentences) - 1:
                position_score = 0.8
            else:
                position_score = 0.6 - (i / len(sentences)) * 0.3
            
            # Length score (moderate length preferred)
            length = len(sentence.split())
            length_score = min(1.0, length / 20.0) * (1.0 - min(1.0, length / 50.0))
            
            # Keyword density score
            words = sentence.lower().split()
            important_words = [w for w in words if len(w) > 4 and w.isalpha()]
            keyword_density = len(important_words) / max(len(words), 1)
            
            # Combine scores
            total_score = (position_score * 0.4 + 
                          length_score * 0.3 + 
                          keyword_density * 0.3)
            
            scored_sentences.append((sentence.strip(), total_score))
        
        # Sort by score and build refined text
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        refined_text = ""
        for sentence, score in scored_sentences:
            if len(refined_text) + len(sentence) + 1 <= max_length:
                refined_text += sentence + " "
            else:
                break
        
        return refined_text.strip()
    
    def process_documents(self, pdf_paths: List[str], persona: str, job: str) -> Dict[str, Any]:
        """Main processing pipeline with comprehensive logging"""
        start_time = time.time()
        
        logger.info(f"Starting document processing pipeline")
        logger.info(f"Documents: {len(pdf_paths)}")
        logger.info(f"Persona: {persona}")
        logger.info(f"Job: {job}")
        
        all_sections = []
        
        # Process each PDF
        for i, pdf_path in enumerate(pdf_paths, 1):
            filename = os.path.basename(pdf_path)
            logger.info(f"Processing document {i}/{len(pdf_paths)}: {filename}")
            
            page_texts = self.extract_text_from_pdf(pdf_path)
            
            if not page_texts:
                logger.warning(f"No text extracted from {filename}")
                continue
            
            document_sections = 0
            for page_num, text in page_texts.items():
                sections = self.segment_into_sections(text, page_num)
                document_sections += len(sections)
                
                for section in sections:
                    relevance_score = self.calculate_relevance_score(
                        section['text'], persona, job
                    )
                    
                    section.update({
                        'document': filename,
                        'relevance_score': relevance_score
                    })
                    
                    all_sections.append(section)
            
            logger.info(f"Extracted {document_sections} sections from {filename}")
        
        if not all_sections:
            logger.error("No sections extracted from any documents")
            raise ValueError("No processable content found in input documents")
        
        logger.info(f"Total sections extracted: {len(all_sections)}")
        
        # Sort by relevance score
        all_sections.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Log top scores for debugging
        top_5_scores = [s['relevance_score'] for s in all_sections[:5]]
        logger.info(f"Top 5 relevance scores: {top_5_scores}")
        
        # Select top sections
        top_sections = all_sections[:10]
        
        # Prepare output structures
        extracted_sections = []
        subsection_analysis = []
        
        for i, section in enumerate(top_sections, 1):
            extracted_sections.append({
                "document": section['document'],
                "page_number": section['page_number'],
                "section_title": section['title'],
                "importance_rank": i
            })
            
            refined_text = self.refine_section_text(section['text'])
            
            subsection_analysis.append({
                "document": section['document'],
                "section_title": section['title'],
                "refined_text": refined_text,
                "page_number": section['page_number']
            })
        
        # Prepare final output
        processing_time = round(time.time() - start_time, 2)
        
        output = {
            "metadata": {
                "input_documents": [os.path.basename(path) for path in pdf_paths],
                "persona": persona,
                "job_to_be_done": job,
                "processing_timestamp": datetime.now().isoformat(),
                "total_sections_analyzed": len(all_sections),
                "system_info": {
                    "python_version": sys.version,
                    "platform": sys.platform
                }
            },
            "extracted_sections": extracted_sections,
            "subsection_analysis": subsection_analysis,
            "processing_time_seconds": processing_time
        }
        
        logger.info(f"Processing completed successfully in {processing_time} seconds")
        return output

def main():
    """Main function optimized for Docker environment"""
    logger.info("=" * 60)
    logger.info("DOCUMENT INTELLIGENCE SYSTEM - DOCKER VERSION")
    logger.info("=" * 60)
    
    parser = argparse.ArgumentParser(description='Document Intelligence System - Docker Version')
    parser.add_argument('--input_dir', 
                       default='/app/input',
                       help='Directory containing PDF files')
    parser.add_argument('--persona', 
                       default=os.getenv('PERSONA', 'Researcher'),
                       help='User persona')
    parser.add_argument('--job', 
                       default=os.getenv('JOB', 'Analyze documents'),
                       help='Job to be done')
    parser.add_argument('--output', 
                       default='/app/output/result.json',
                       help='Output JSON file')
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        logger.error(f"Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    
    # Get PDF files
    pdf_files = []
    for f in os.listdir(args.input_dir):
        if f.lower().endswith('.pdf'):
            pdf_files.append(os.path.join(args.input_dir, f))
    
    if not pdf_files:
        logger.error(f"No PDF files found in {args.input_dir}")
        logger.info("Please ensure PDF files are mounted to /app/input in the container")
        sys.exit(1)
    
    logger.info(f"Found {len(pdf_files)} PDF files")
    
    # Create output directory
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Initialize system
        system = DocumentIntelligenceSystem()
        
        # Process documents
        result = system.process_documents(pdf_files, args.persona, args.job)
        
        # Save output with proper encoding
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to: {args.output}")
        logger.info(f"Processing summary:")
        logger.info(f"  - Documents processed: {len(pdf_files)}")
        logger.info(f"  - Sections analyzed: {result['metadata']['total_sections_analyzed']}")
        logger.info(f"  - Top sections selected: {len(result['extracted_sections'])}")
        logger.info(f"  - Processing time: {result['processing_time_seconds']} seconds")
        
        # Return success code
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()