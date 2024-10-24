FROM python:3.8-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Install additional dependencies or tools if needed
RUN apt-get update && apt-get install -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

CMD ["sh", "-c", "python app/ingestion/ingest_pdfs.py && streamlit run app/main.py"]
