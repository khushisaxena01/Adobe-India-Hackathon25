#!/usr/bin/env python3
"""
Document Intelligence System for Adobe Hackathon 1B
Extracts and prioritizes relevant sections based on persona and job-to-be-done
"""

import json
import os
import re
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple
import PyPDF2
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import argparse

class DocumentIntelligenceSystem:
    def __init__(self):
        """Initialize the document intelligence system"""
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        self.stop_words = set(stopwords.words('english'))
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> Dict[int, str]:
        """Extract text from PDF, page by page"""
        page_texts = {}
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    text = page.extract_text()
                    if text.strip():
                        page_texts[page_num] = text
                        
        except Exception as e:
            print(f"Error reading PDF {pdf_path}: {e}")
            
        return page_texts
    
    def segment_into_sections(self, text: str, page_num: int) -> List[Dict[str, Any]]:
        """Segment text into logical sections"""
        sections = []
        
        # Split by common section headers
        section_patterns = [
            r'\n\s*\d+\.\s+[A-Z][^\.]*\n',  # Numbered sections
            r'\n\s*[A-Z][A-Z\s]{3,}\n',     # ALL CAPS headers
            r'\n\s*[A-Z][^\.]{10,}\n',      # Title case headers
            r'\n\s*Abstract\s*\n',
            r'\n\s*Introduction\s*\n',
            r'\n\s*Methodology\s*\n',
            r'\n\s*Results\s*\n',
            r'\n\s*Discussion\s*\n',
            r'\n\s*Conclusion\s*\n',
            r'\n\s*References\s*\n'
        ]
        
        # Find section breaks
        breaks = [0]
        section_titles = ["Beginning"]
        
        for pattern in section_patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            for match in matches:
                title = match.group().strip()
                if len(title) > 0:
                    breaks.append(match.start())
                    section_titles.append(title)
        
        breaks.append(len(text))
        
        # Create sections
        for i in range(len(breaks) - 1):
            start = breaks[i]
            end = breaks[i + 1]
            section_text = text[start:end].strip()
            
            if len(section_text) > 50:  # Only include substantial sections
                sections.append({
                    'title': section_titles[i] if i < len(section_titles) else f"Section {i}",
                    'text': section_text,
                    'page_number': page_num,
                    'word_count': len(section_text.split())
                })
        
        return sections
    
    def calculate_relevance_score(self, section_text: str, persona: str, job: str) -> float:
        """Calculate relevance score based on persona and job"""
        
        # Create query from persona and job
        query = f"{persona} {job}".lower()
        
        # Extract keywords from persona and job
        persona_keywords = self.extract_keywords(persona.lower())
        job_keywords = self.extract_keywords(job.lower())
        
        section_lower = section_text.lower()
        
        # Keyword matching score
        keyword_score = 0
        for keyword in persona_keywords + job_keywords:
            if keyword in section_lower:
                keyword_score += 1
        
        # TF-IDF similarity score
        try:
            corpus = [query, section_text.lower()]
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
            similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except:
            similarity_score = 0
        
        # Combine scores
        final_score = (keyword_score * 0.4) + (similarity_score * 0.6)
        
        return final_score
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text"""
        words = word_tokenize(text.lower())
        keywords = [word for word in words 
                   if word.isalpha() and 
                   word not in self.stop_words and 
                   len(word) > 3]
        return keywords
    
    def refine_section_text(self, text: str, max_length: int = 500) -> str:
        """Refine and summarize section text"""
        sentences = sent_tokenize(text)
        
        if len(text) <= max_length:
            return text.strip()
        
        # Score sentences by position and keyword density
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            # Position score (earlier sentences get higher score)
            position_score = 1.0 - (i / len(sentences)) * 0.5
            
            # Keyword density score
            words = len(sentence.split())
            keyword_score = len([w for w in sentence.lower().split() 
                               if w not in self.stop_words]) / max(words, 1)
            
            total_score = position_score + keyword_score
            scored_sentences.append((sentence, total_score))
        
        # Sort by score and select top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        refined_text = ""
        for sentence, _ in scored_sentences:
            if len(refined_text) + len(sentence) <= max_length:
                refined_text += sentence + " "
            else:
                break
        
        return refined_text.strip()
    
    def process_documents(self, pdf_paths: List[str], persona: str, job: str) -> Dict[str, Any]:
        """Main processing function"""
        start_time = time.time()
        
        all_sections = []
        
        # Process each PDF
        for pdf_path in pdf_paths:
            print(f"Processing: {os.path.basename(pdf_path)}")
            
            page_texts = self.extract_text_from_pdf(pdf_path)
            
            for page_num, text in page_texts.items():
                sections = self.segment_into_sections(text, page_num)
                
                for section in sections:
                    relevance_score = self.calculate_relevance_score(
                        section['text'], persona, job
                    )
                    
                    section.update({
                        'document': os.path.basename(pdf_path),
                        'relevance_score': relevance_score
                    })
                    
                    all_sections.append(section)
        
        # Sort by relevance score
        all_sections.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Select top sections
        top_sections = all_sections[:10]  # Top 10 most relevant sections
        
        # Prepare output
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
        output = {
            "metadata": {
                "input_documents": [os.path.basename(path) for path in pdf_paths],
                "persona": persona,
                "job_to_be_done": job,
                "processing_timestamp": datetime.now().isoformat()
            },
            "extracted_sections": extracted_sections,
            "subsection_analysis": subsection_analysis,
            "processing_time_seconds": round(time.time() - start_time, 2)
        }
        
        return output

def main():
    parser = argparse.ArgumentParser(description='Document Intelligence System')
    parser.add_argument('--input_dir', required=True, help='Directory containing PDF files')
    parser.add_argument('--persona', required=True, help='User persona')
    parser.add_argument('--job', required=True, help='Job to be done')
    parser.add_argument('--output', default='output.json', help='Output JSON file')
    
    args = parser.parse_args()
    
    # Get PDF files
    pdf_files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) 
                 if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print("No PDF files found in the input directory")
        return
    
    print(f"Found {len(pdf_files)} PDF files")
    print(f"Persona: {args.persona}")
    print(f"Job: {args.job}")
    
    # Initialize system
    system = DocumentIntelligenceSystem()
    
    # Process documents
    result = system.process_documents(pdf_files, args.persona, args.job)
    
    # Save output
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Processing completed in {result['processing_time_seconds']} seconds")
    print(f"Output saved to {args.output}")

if __name__ == "__main__":
    main()