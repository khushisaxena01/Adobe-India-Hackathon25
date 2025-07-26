"""
Document Intelligence System for Adobe Hackathon 1B - Auto Mode
Automatically processes PDFs from 'input' folder and saves to 'output' folder
Shows results in terminal and saves JSON output
"""

import json
import os
import re
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple
import argparse
import sys

# Try importing required libraries with helpful error messages
try:
    import PyPDF2
except ImportError:
    print("PyPDF2 not found. Please install: pip install PyPDF2==3.0.1")
    sys.exit(1)

try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
except ImportError:
    print("NLTK not found. Please install: pip install nltk==3.8.1")
    sys.exit(1)

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
except ImportError:
    print("Scikit-learn not found. Please install: pip install scikit-learn==1.3.2 numpy==1.24.3")
    sys.exit(1)

class DocumentIntelligenceSystem:
    def __init__(self):
        """Initialize the document intelligence system with Windows-specific handling"""
        print("🚀 Initializing Document Intelligence System...")
        
        # Download NLTK data with progress indication
        self._download_nltk_data()
        
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            print("📥 Downloading stopwords...")
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words('english'))
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        print("✅ System initialized successfully!\n")
    
    def _download_nltk_data(self):
        """Download required NLTK data with Windows-friendly paths"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("📥 Downloading NLTK punkt tokenizer...")
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            print("📥 Downloading NLTK stopwords...")
            nltk.download('stopwords', quiet=True)
    
    def extract_text_from_pdf(self, pdf_path: str) -> Dict[int, str]:
        """Extract text from PDF with Windows path handling"""
        page_texts = {}
        
        try:
            # Use raw string for Windows paths
            pdf_path = os.path.normpath(pdf_path)
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                print(f"  📄 Extracting from {len(pdf_reader.pages)} pages...")
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    text = page.extract_text()
                    if text.strip():
                        page_texts[page_num] = text
                        
        except Exception as e:
            print(f"❌ Error reading PDF {pdf_path}: {e}")
            
        return page_texts
    
    def segment_into_sections(self, text: str, page_num: int) -> List[Dict[str, Any]]:
        """Segment text into logical sections"""
        sections = []
        
        # Split by common section headers (Windows line endings compatible)
        section_patterns = [
            r'(?:\r?\n)\s*\d+\.\s+[A-Z][^\.\r\n]*(?:\r?\n)',  # Numbered sections
            r'(?:\r?\n)\s*[A-Z][A-Z\s]{3,}(?:\r?\n)',         # ALL CAPS headers
            r'(?:\r?\n)\s*[A-Z][^\.\r\n]{10,}(?:\r?\n)',      # Title case headers
            r'(?:\r?\n)\s*Abstract\s*(?:\r?\n)',
            r'(?:\r?\n)\s*Introduction\s*(?:\r?\n)',
            r'(?:\r?\n)\s*Methodology\s*(?:\r?\n)',
            r'(?:\r?\n)\s*Results\s*(?:\r?\n)',
            r'(?:\r?\n)\s*Discussion\s*(?:\r?\n)',
            r'(?:\r?\n)\s*Conclusion\s*(?:\r?\n)',
            r'(?:\r?\n)\s*References\s*(?:\r?\n)'
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
        try:
            words = word_tokenize(text.lower())
            keywords = [word for word in words 
                       if word.isalpha() and 
                       word not in self.stop_words and 
                       len(word) > 3]
            return keywords
        except:
            # Fallback for tokenization issues
            words = text.lower().split()
            keywords = [word for word in words 
                       if word.isalpha() and len(word) > 3]
            return keywords
    
    def refine_section_text(self, text: str, max_length: int = 500) -> str:
        """Refine and summarize section text"""
        try:
            sentences = sent_tokenize(text)
        except:
            # Fallback sentence splitting
            sentences = text.split('. ')
        
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
                               if len(w) > 3]) / max(words, 1)
            
            total_score = position_score + keyword_score
            scored_sentences.append((sentence, total_score))
        
        # Sort by score and select top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        refined_text = ""
        for sentence, _ in scored_sentences:
            if len(refined_text) + len(sentence) <= max_length:
                refined_text += sentence + ". "
            else:
                break
        
        return refined_text.strip()
    
    def process_documents(self, pdf_paths: List[str], persona: str, job: str) -> Dict[str, Any]:
        """Main processing function with progress indication"""
        start_time = time.time()
        
        print(f"📊 Processing {len(pdf_paths)} documents...")
        print(f"👤 Persona: {persona}")
        print(f"🎯 Job: {job}\n")
        
        all_sections = []
        
        # Process each PDF
        for i, pdf_path in enumerate(pdf_paths, 1):
            filename = os.path.basename(pdf_path)
            print(f"[{i}/{len(pdf_paths)}] 📖 Processing: {filename}")
            
            page_texts = self.extract_text_from_pdf(pdf_path)
            
            for page_num, text in page_texts.items():
                sections = self.segment_into_sections(text, page_num)
                
                for section in sections:
                    relevance_score = self.calculate_relevance_score(
                        section['text'], persona, job
                    )
                    
                    section.update({
                        'document': filename,
                        'relevance_score': relevance_score
                    })
                    
                    all_sections.append(section)
        
        print(f"\n🔍 Found {len(all_sections)} total sections. Ranking by relevance...")
        
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
                "importance_rank": i,
                "relevance_score": round(section['relevance_score'], 3)
            })
            
            refined_text = self.refine_section_text(section['text'])
            
            subsection_analysis.append({
                "document": section['document'],
                "section_title": section['title'],
                "refined_text": refined_text,
                "page_number": section['page_number'],
                "relevance_score": round(section['relevance_score'], 3)
            })
        
        # Prepare final output
        output = {
            "metadata": {
                "input_documents": [os.path.basename(path) for path in pdf_paths],
                "persona": persona,
                "job_to_be_done": job,
                "processing_timestamp": datetime.now().isoformat(),
                "total_sections_found": len(all_sections),
                "top_sections_selected": len(top_sections)
            },
            "extracted_sections": extracted_sections,
            "subsection_analysis": subsection_analysis,
            "processing_time_seconds": round(time.time() - start_time, 2)
        }
        
        print(f"✅ Processing completed in {output['processing_time_seconds']} seconds")
        
        return output
    
    def display_results_in_terminal(self, result: Dict[str, Any]):
        """Display results in a nice terminal format"""
        print("\n" + "="*80)
        print("🎯 DOCUMENT INTELLIGENCE RESULTS")
        print("="*80)
        
        # Metadata
        metadata = result['metadata']
        print(f"📂 Documents processed: {', '.join(metadata['input_documents'])}")
        print(f"👤 Persona: {metadata['persona']}")
        print(f"🎯 Job: {metadata['job_to_be_done']}")
        print(f"⏱️  Processing time: {result['processing_time_seconds']} seconds")
        print(f"📊 Total sections found: {metadata['total_sections_found']}")
        print(f"🏆 Top sections selected: {metadata['top_sections_selected']}")
        
        print("\n" + "="*80)
        print("🏆 TOP RELEVANT SECTIONS")
        print("="*80)
        
        for i, section in enumerate(result['subsection_analysis'], 1):
            print(f"\n📍 RANK {i} (Score: {section['relevance_score']})")
            print(f"📄 Document: {section['document']}")
            print(f"📖 Section: {section['section_title']}")
            print(f"📄 Page: {section['page_number']}")
            print(f"📝 Content Preview:")
            print("-" * 60)
            # Wrap text nicely
            text = section['refined_text']
            words = text.split()
            lines = []
            current_line = []
            current_length = 0
            
            for word in words:
                if current_length + len(word) + 1 > 75:  # 75 chars per line
                    lines.append(" ".join(current_line))
                    current_line = [word]
                    current_length = len(word)
                else:
                    current_line.append(word)
                    current_length += len(word) + 1
            
            if current_line:
                lines.append(" ".join(current_line))
            
            for line in lines:
                print(f"   {line}")
            
            print("-" * 60)
        
        print("\n" + "="*80)

def setup_directories():
    """Setup input and output directories"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, 'input')
    output_dir = os.path.join(script_dir, 'output')
    
    # Create directories if they don't exist
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    return input_dir, output_dir

def get_user_preferences():
    """Get user preferences for persona and job"""
    print("🤖 DOCUMENT INTELLIGENCE SYSTEM - AUTO MODE")
    print("="*60)
    print("📁 This system automatically processes PDFs from the 'input' folder")
    print("💾 Results are saved to the 'output' folder\n")
    
    # Common examples to help users
    print("💡 Persona Examples:")
    print("   • data scientist, researcher, student, marketing manager")
    print("   • business analyst, software engineer, product manager")
    
    print("\n💡 Job Examples:")
    print("   • extract methodology sections")
    print("   • find key insights and conclusions")
    print("   • summarize main findings")
    print("   • identify technical specifications")
    print("   • locate market research data\n")
    
    # Get input with defaults
    persona = input("👤 Enter your persona (or press Enter for 'researcher'): ").strip()
    if not persona:
        persona = "researcher"
    
    job = input("🎯 Enter the job to be done (or press Enter for 'extract key insights'): ").strip()
    if not job:
        job = "extract key insights"
    
    return persona, job

def main():
    """Main function with auto-processing"""
    try:
        # Setup directories
        input_dir, output_dir = setup_directories()
        
        # Check for PDF files in input directory
        pdf_files = []
        if os.path.exists(input_dir):
            for f in os.listdir(input_dir):
                if f.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(input_dir, f))
        
        if not pdf_files:
            print("❌ No PDF files found in the 'input' folder!")
            print(f"📁 Please place your PDF files in: {input_dir}")
            print("🔄 Then run this script again.")
            input("\nPress Enter to exit...")
            return
        
        print(f"✅ Found {len(pdf_files)} PDF file(s):")
        for pdf_file in pdf_files:
            print(f"   📄 {os.path.basename(pdf_file)}")
        print()
        
        # Get user preferences
        persona, job = get_user_preferences()
        
        print(f"\n🚀 Starting processing...")
        print(f"👤 Persona: {persona}")
        print(f"🎯 Job: {job}")
        
        # Initialize system
        system = DocumentIntelligenceSystem()
        
        # Process documents
        result = system.process_documents(pdf_files, persona, job)
        
        # Display results in terminal
        system.display_results_in_terminal(result)
        
        # Save JSON output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"analysis_results_{timestamp}.json")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Detailed results saved to: {output_file}")
        print(f"📊 Summary: Found {len(result['extracted_sections'])} most relevant sections")
        
        print("\n🎉 Processing completed successfully!")
        input("\nPress Enter to exit...")
        
    except KeyboardInterrupt:
        print("\n⏹️  Processing interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()