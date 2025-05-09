import os
import requests
from supabase import create_client
from src.utils import (
    create_embedding,
    add_documents_to_supabase,
    get_supabase_client
)

# Initialize Supabase client
supabase = get_supabase_client()

# URL to crawl
url = "https://www.simoahava.com/analytics/create-facebook-pixel-custom-tag-template/"

# Get the page content
response = requests.get(url)
response.raise_for_status()
content = response.text

# Split into chunks (using a simple split for now)
chunks = [content[i:i+1000] for i in range(0, len(content), 1000)]

# Create metadata
metadatas = [{"source": "simoahava_docs"}] * len(chunks)

# Create chunk numbers
chunk_numbers = list(range(len(chunks)))

# Create URL list
urls = [url] * len(chunks)

# Create URL to full document mapping
url_to_full_document = {url: content}

# Add documents to Supabase
add_documents_to_supabase(
    supabase,
    urls,
    chunk_numbers,
    chunks,
    metadatas,
    url_to_full_document
)

print("Done!")
