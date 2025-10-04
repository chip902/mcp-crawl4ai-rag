#!/usr/bin/env python3
"""
OpenAPI-compatible REST API server for Open-WebUI
Exposes the same crawling functionality as REST endpoints
"""

import os
import sys
import logging
import json
import asyncio
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from dotenv import load_dotenv
import uvicorn

# Import existing utilities
from utils import (
    validate_ollama_connection,
    validate_supabase_connection,
    get_supabase_client,
    add_documents_to_supabase,
    search_documents,
)
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

# Load environment
project_root = Path(__file__).resolve().parent.parent
dotenv_path = project_root / '.env'
load_dotenv(dotenv_path, override=True)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Global instances
supabase_client = None
crawler = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown"""
    global supabase_client, crawler

    # Startup
    logger.info("Starting OpenAPI server...")
    try:
        # Initialize Supabase client
        supabase_client = get_supabase_client()
        if not validate_supabase_connection(supabase_client):
            logger.error("Supabase connection validation failed")
            raise RuntimeError("Supabase connection failed")

        # Validate Ollama
        if not validate_ollama_connection():
            logger.warning("Ollama connection validation failed - embeddings may not work")

        # Initialize crawler
        browser_config = BrowserConfig(
            headless=True,
            verbose=False
        )
        crawler = AsyncWebCrawler(config=browser_config)
        await crawler.__aenter__()

        logger.info("All connections validated successfully")
        logger.info(f"OpenAPI docs available at: http://0.0.0.0:8082/docs")
        logger.info(f"OpenAPI spec available at: http://0.0.0.0:8082/openapi.json")

    except Exception as e:
        logger.error(f"Startup validation failed: {e}")
        raise

    yield

    # Shutdown
    if crawler:
        await crawler.__aexit__(None, None, None)
        logger.info("Crawler closed")


# FastAPI app with lifespan
app = FastAPI(
    title="Crawl4AI RAG API",
    description="Web crawling and RAG query API for documentation knowledge base",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# CORS middleware for Open-WebUI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class SmartCrawlRequest(BaseModel):
    url: str
    max_depth: int = 3
    max_concurrent: int = 10
    chunk_size: int = 5000

class CrawlSinglePageRequest(BaseModel):
    url: str

class RAGQueryRequest(BaseModel):
    query: str
    source: Optional[str] = None
    match_count: int = 5

class ScanGitHubRequest(BaseModel):
    repo_owner: str
    repo_name: str

class CrawlResponse(BaseModel):
    success: bool
    message: str
    details: dict

class RAGResponse(BaseModel):
    success: bool
    query: str
    source_filter: Optional[str]
    results: List[dict]
    count: int

class SourcesResponse(BaseModel):
    success: bool
    sources: List[str]
    count: int


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Crawl4AI RAG API",
        "status": "running",
        "docs": "/docs",
        "openapi": "/openapi.json"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.post("/crawl/smart", response_model=CrawlResponse, tags=["Crawling"])
async def smart_crawl(request: SmartCrawlRequest):
    """
    Intelligently crawl a URL based on its type.

    Automatically detects:
    - Sitemaps (sitemap.xml) - crawls all URLs
    - Text files (llms.txt) - directly retrieves content
    - Regular webpages - recursively crawls up to max_depth

    All content is chunked and stored in Supabase for RAG queries.
    """
    try:
        # TODO: Implement smart crawling logic from MCP server
        # For now, just crawl the single page
        return await crawl_single(CrawlSinglePageRequest(url=request.url))
    except Exception as e:
        logger.error(f"Smart crawl error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/crawl/single", response_model=CrawlResponse, tags=["Crawling"])
async def crawl_single(request: CrawlSinglePageRequest):
    """
    Crawl a single web page without following links.

    Ideal for quickly retrieving content from a specific URL.
    Content is stored in Supabase for later retrieval.
    """
    try:
        if not crawler:
            raise HTTPException(status_code=500, detail="Crawler not initialized")

        # Configure crawl with timeout
        run_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            stream=False,
            page_timeout=60000,
            wait_until="networkidle"
        )

        # Crawl the page
        logger.info(f"Crawling URL: {request.url}")
        result = await asyncio.wait_for(
            crawler.arun(url=request.url, config=run_config),
            timeout=90.0
        )

        if result.success and result.markdown:
            from urllib.parse import urlparse
            source_domain = urlparse(request.url).netloc

            # Store in Supabase
            add_documents_to_supabase(
                client=supabase_client,
                urls=[request.url],
                contents=[result.markdown],
                metadatas=[{
                    "source": source_domain,
                    "crawl_type": "single_page"
                }]
            )

            return CrawlResponse(
                success=True,
                message="Page crawled successfully",
                details={
                    "url": request.url,
                    "content_length": len(result.markdown),
                    "stored": True
                }
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to crawl page")

    except Exception as e:
        logger.error(f"Single page crawl error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=RAGResponse, tags=["Query"])
async def rag_query(request: RAGQueryRequest):
    """
    Perform a RAG query on the stored documentation.

    Searches the vector database for content relevant to your query.
    Optionally filter by source domain.
    """
    try:
        if not supabase_client:
            raise HTTPException(status_code=500, detail="Supabase client not initialized")

        # Prepare filter
        filter_metadata = None
        if request.source and request.source.strip():
            filter_metadata = {"source": request.source}

        # Perform search
        results = search_documents(
            client=supabase_client,
            query=request.query,
            match_count=request.match_count,
            filter_metadata=filter_metadata
        )

        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "url": result.get("url"),
                "content": result.get("content"),
                "metadata": result.get("metadata"),
                "similarity": result.get("similarity")
            })

        return RAGResponse(
            success=True,
            query=request.query,
            source_filter=request.source,
            results=formatted_results,
            count=len(formatted_results)
        )
    except Exception as e:
        logger.error(f"RAG query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sources", response_model=SourcesResponse, tags=["Query"])
async def get_sources():
    """
    Get all available sources in the knowledge base.

    Returns a list of unique source domains that have been crawled.
    """
    try:
        if not supabase_client:
            raise HTTPException(status_code=500, detail="Supabase client not initialized")

        # Query for unique sources
        result = supabase_client.from_('crawled_pages')\
            .select('metadata')\
            .not_.is_('metadata->>source', 'null')\
            .execute()

        # Extract unique sources
        unique_sources = set()
        if result.data:
            for item in result.data:
                source = item.get('metadata', {}).get('source')
                if source:
                    unique_sources.add(source)

        # Convert to sorted list
        sources = sorted(list(unique_sources))

        return SourcesResponse(
            success=True,
            sources=sources,
            count=len(sources)
        )
    except Exception as e:
        logger.error(f"Get sources error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/github/docs", response_model=CrawlResponse, tags=["GitHub"])
async def scan_github_docs(request: ScanGitHubRequest):
    """
    Scan a GitHub repository for markdown documentation files.

    Extracts and stores all .md files in Supabase.
    """
    try:
        # TODO: Implement GitHub scanning from MCP server
        raise HTTPException(status_code=501, detail="GitHub docs scanning not yet implemented in REST API")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"GitHub docs scan error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/github/source", response_model=CrawlResponse, tags=["GitHub"])
async def scan_github_source(request: ScanGitHubRequest):
    """
    Scan a GitHub repository for source code files.

    Extracts implementation details, API usage examples, and patterns.
    """
    try:
        # TODO: Implement GitHub source scanning from MCP server
        raise HTTPException(status_code=501, detail="GitHub source scanning not yet implemented in REST API")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"GitHub source scan error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def main():
    """Run the OpenAPI server"""
    port = int(os.getenv("OPENAPI_PORT", "8081"))
    host = os.getenv("HOST", "0.0.0.0")

    logger.info(f"Starting OpenAPI server on {host}:{port}")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    main()
