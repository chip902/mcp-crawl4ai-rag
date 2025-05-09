import os
from supabase import create_client
from src.utils import search_documents

# Initialize Supabase client
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# Search for documents
results = search_documents(
    supabase,
    "analytics best practices",
    match_count=3
)

# Print results
for result in results:
    print(f"\nURL: {result.get('url')}")
    print(f"Title: {result.get('title')}")
    print(f"Summary: {result.get('summary')}")
    print("---")
