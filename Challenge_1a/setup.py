# This setup script checks if PyMuPDF is installed, creates input/output directories, and prepares your 
# environment for extracting PDF table of contents/outlines.

#!/usr/bin/env python3
"""
Setup script to prepare the environment for PDF outline extraction.
"""

import os
import sys

def check_dependencies():
    """Check if required packages are installed."""
    try:
        import fitz
        print("✓ PyMuPDF is installed")
        return True
    except ImportError:
        print("✗ PyMuPDF not found")
        print("Please install it with: pip install PyMuPDF")
        return False

def create_directories():
    """Create input and output directories."""
    directories = ['input', 'output']
    
    for dir_name in directories:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"✓ Created {dir_name}/ directory")
        else:
            print(f"✓ {dir_name}/ directory already exists")

def main():
    """Main setup function."""
    print("PDF Outline Extractor Setup")
    print("=" * 30)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    print("\nSetup complete!")
    print("\nNext steps:")
    print("1. Place your PDF files in the input/ directory")
    print("2. Run: python round_A.py")
    print("3. Check the output/ directory for JSON results")
    
    # Check if there are any PDFs in input
    if os.path.exists('input'):
        pdf_files = [f for f in os.listdir('input') if f.lower().endswith('.pdf')]
        if pdf_files:
            print(f"\nFound {len(pdf_files)} PDF file(s) in input/ directory:")
            for pdf_file in pdf_files:
                print(f"  - {pdf_file}")
        else:
            print("\nNo PDF files found in input/ directory yet.")

if __name__ == "__main__":
    main()