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
