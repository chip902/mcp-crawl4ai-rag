"""
MCP server for web crawling with Crawl4AI.

This server provides tools to crawl websites using Crawl4AI, automatically detecting
the appropriate crawl method based on URL type (sitemap, txt file, or regular webpage).
"""
from mcp.server.fastmcp import FastMCP, Context
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import List, Dict, Any
from urllib.parse import urlparse, urldefrag
from xml.etree import ElementTree
from dotenv import load_dotenv
from supabase import Client
from pathlib import Path
import requests
import asyncio
import json
import os
import re

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode, MemoryAdaptiveDispatcher
from utils import get_supabase_client, add_documents_to_supabase, search_documents
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging
import sys

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables from the project root .env file
project_root = Path(__file__).resolve().parent.parent
dotenv_path = project_root / '.env'

# Force override of existing environment variables
load_dotenv(dotenv_path, override=True)


def validate_configuration():
    """
    Validate required environment variables and system dependencies on startup.
    Raises ValueError if critical configuration is missing.
    """
    logger.info("Validating configuration...")

    required_vars = {
        "SUPABASE_URL": os.getenv("SUPABASE_URL"),
        "SUPABASE_SERVICE_KEY": os.getenv("SUPABASE_SERVICE_KEY"),
    }

    optional_vars = {
        "TRANSPORT": os.getenv("TRANSPORT", "sse"),
        "HOST": os.getenv("HOST", "0.0.0.0"),
        "PORT": os.getenv("PORT", "8051"),
        "OLLAMA_BASE_URL": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        "OLLAMA_MODEL": os.getenv("OLLAMA_MODEL", "nomic-embed-text"),
    }

    # Check required variables
    missing_vars = [name for name, value in required_vars.items() if not value]
    if missing_vars:
        error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Log configuration
    logger.info("Configuration validated successfully:")
    for name, value in required_vars.items():
        # Don't log sensitive values in full
        if "KEY" in name or "SECRET" in name:
            logger.info(f"  {name}: {'*' * 20}")
        else:
            logger.info(f"  {name}: {value}")

    for name, value in optional_vars.items():
        logger.info(f"  {name}: {value}")

    # Validate Ollama connection (non-blocking)
    from utils import validate_ollama_connection
    ollama_ok = validate_ollama_connection()
    if not ollama_ok:
        logger.warning("Ollama is not available - will use fallback embeddings")
    else:
        logger.info("Ollama connection validated")

    logger.info("Configuration validation complete")


# Validate configuration on module load
try:
    validate_configuration()
except Exception as e:
    logger.error(f"Configuration validation failed: {e}")
    logger.error("Server may not function correctly. Please check your .env file.")
    # Don't raise here - let the server start but log the error


# Initialize FastAPI app first (needed for MCP initialization)
app = FastAPI(
    title="Crawl4AI MCP Server",
    description="A server for web crawling and document processing using Crawl4AI",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Create a dataclass for our application context
@dataclass
class Crawl4AIContext:
    """Context for the Crawl4AI MCP server."""
    crawler: AsyncWebCrawler
    supabase_client: Client


@asynccontextmanager
async def crawl4ai_lifespan(server: FastMCP) -> AsyncIterator[Crawl4AIContext]:
    """
    Manages the Crawl4AI client lifecycle.


    Args:
        server: The FastMCP server instance


    Yields:
        Crawl4AIContext: The context containing the Crawl4AI crawler and Supabase client
    """
    # Create browser configuration
    browser_config = BrowserConfig(
        headless=True,
        verbose=False
    )

    # Initialize the crawler
    crawler = AsyncWebCrawler(config=browser_config)
    try:
        await crawler.__aenter__()
    except Exception as exc:
        logging.error(f"Error initializing crawler: {exc}")
        raise

    # Initialize Supabase client
    try:
        supabase_client = get_supabase_client()
    except Exception as exc:
        logging.error(f"Error initializing Supabase client: {exc}")
        await crawler.__aexit__(None, None, None)
        raise

    try:
        yield Crawl4AIContext(
            crawler=crawler,
            supabase_client=supabase_client
        )
    finally:
        # Clean up the crawler
        try:
            await crawler.__aexit__(None, None, None)
        except Exception as exc:
            logging.error(f"Error cleaning up crawler: {exc}")


# Initialize FastMCP server with custom app
mcp = FastMCP(
    "mcp-crawl4ai-rag",
    description="MCP server for RAG and web crawling with Crawl4AI",
    lifespan=crawl4ai_lifespan,
    host=os.getenv("HOST", "0.0.0.0"),
    port=int(os.getenv("PORT", "8051")),
    app=app  # Use our FastAPI app with health endpoint
)


def is_sitemap(url: str) -> bool:
    """
    Check if a URL is a sitemap.


    Args:
        url: URL to check


    Returns:
        True if the URL is a sitemap, False otherwise
    """
    return url.endswith('sitemap.xml') or 'sitemap' in urlparse(url).path


def is_txt(url: str) -> bool:
    """
    Check if a URL is a text file.


    Args:
        url: URL to check


    Returns:
        True if the URL is a text file, False otherwise
    """
    return url.endswith('.txt')


def parse_sitemap(sitemap_url: str) -> List[str]:
    """
    Parse a sitemap and extract URLs.


    Args:
        sitemap_url: URL of the sitemap


    Returns:
        List of URLs found in the sitemap
    """
    resp = requests.get(sitemap_url)
    urls = []

    if resp.status_code == 200:
        try:
            tree = ElementTree.fromstring(resp.content)
            urls = [loc.text for loc in tree.findall('.//{*}loc')]
        except Exception as e:
            print(f"Error parsing sitemap XML: {e}")

    return urls


def smart_chunk_markdown(text: str, chunk_size: int = 5000) -> List[str]:
    """Split text into chunks, respecting code blocks and paragraphs."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        # Calculate end position
        end = start + chunk_size

        # If we're at the end of the text, just take what's left
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        # Try to find a code block boundary first (```)
        chunk = text[start:end]
        code_block = chunk.rfind('```')
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block

        # If no code block, try to break at a paragraph
        elif '\n\n' in chunk:
            # Find the last paragraph break
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_break

        # If no paragraph break, try to break at a sentence
        elif '. ' in chunk:
            # Find the last sentence break
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_period + 1

        # Extract chunk and clean it up
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position for next chunk
        start = end

    return chunks


def extract_section_info(chunk: str) -> Dict[str, Any]:
    """
    Extracts headers and stats from a chunk.


    Args:
        chunk: Markdown chunk


    Returns:
        Dictionary with headers and stats
    """
    headers = re.findall(r'^(#+)\s+(.+)$', chunk, re.MULTILINE)
    header_str = '; '.join(
        [f'{h[0]} {h[1]}' for h in headers]) if headers else ''
    header_str = '; '.join(
        [f'{h[0]} {h[1]}' for h in headers]) if headers else ''

    return {
        "headers": header_str,
        "char_count": len(chunk),
        "word_count": len(chunk.split())
    }


@mcp.tool()
async def crawl_single_page(ctx: Context, url: str) -> str:
    """
    Crawl a single web page and store its content in Supabase.


    This tool is ideal for quickly retrieving content from a specific URL without following links.
    The content is stored in Supabase for later retrieval and querying.


    Args:
        ctx: The MCP server provided context
        url: URL of the web page to crawl


    Returns:
        Summary of the crawling operation and storage in Supabase
    """
    try:
        # Get the crawler from the context
        crawler = ctx.request_context.lifespan_context.crawler
        supabase_client = ctx.request_context.lifespan_context.supabase_client

        # Configure the crawl with timeout
        run_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            stream=False,
            page_timeout=60000,  # 60 second timeout
            wait_until="networkidle"
        )

        # Crawl the page with timeout
        logger.info(f"Starting crawl for URL: {url}")
        result = await asyncio.wait_for(
            crawler.arun(url=url, config=run_config),
            timeout=90.0  # 90 second overall timeout
        )

        if result.success and result.markdown:
            # Chunk the content
            chunks = smart_chunk_markdown(result.markdown)

            # Prepare data for Supabase
            urls = []
            chunk_numbers = []
            contents = []
            metadatas = []

            for i, chunk in enumerate(chunks):
                urls.append(url)
                chunk_numbers.append(i)
                contents.append(chunk)

                # Extract metadata
                meta = extract_section_info(chunk)
                meta["chunk_index"] = i
                meta["url"] = url
                meta["source"] = urlparse(url).netloc
                meta["crawl_time"] = str(
                    asyncio.current_task().get_coro().__name__)
                meta["crawl_time"] = str(
                    asyncio.current_task().get_coro().__name__)
                metadatas.append(meta)

            # Create url_to_full_document mapping
            url_to_full_document = {url: result.markdown}

            # Add to Supabase
            add_documents_to_supabase(
                supabase_client, urls, chunk_numbers, contents, metadatas, url_to_full_document)

            return json.dumps({
                "success": True,
                "url": url,
                "chunks_stored": len(chunks),
                "content_length": len(result.markdown),
                "links_count": {
                    "internal": len(result.links.get("internal", [])),
                    "external": len(result.links.get("external", []))
                }
            }, indent=2)
        else:
            logger.warning(f"Crawl failed for {url}: {result.error_message}")
            return json.dumps({
                "success": False,
                "url": url,
                "error": result.error_message
            }, indent=2)
    except asyncio.TimeoutError:
        logger.error(f"Timeout crawling {url} after 90 seconds")
        return json.dumps({
            "success": False,
            "url": url,
            "error": "Crawl operation timed out after 90 seconds"
        }, indent=2)
    except Exception as e:
        logger.error(f"Error crawling {url}: {str(e)}", exc_info=True)
        return json.dumps({
            "success": False,
            "url": url,
            "error": str(e),
            "error_type": type(e).__name__
        }, indent=2)


@mcp.tool()
async def smart_crawl_url(ctx: Context, url: str, max_depth: int = 3, max_concurrent: int = 10, chunk_size: int = 5000) -> str:
    """
    Intelligently crawl a URL based on its type and store content in Supabase.


    This tool automatically detects the URL type and applies the appropriate crawling method:
    - For sitemaps: Extracts and crawls all URLs in parallel
    - For text files (llms.txt): Directly retrieves the content
    - For regular webpages: Recursively crawls internal links up to the specified depth


    All crawled content is chunked and stored in Supabase for later retrieval and querying.


    Args:
        ctx: The MCP server provided context
        url: URL to crawl (can be a regular webpage, sitemap.xml, or .txt file)
        max_depth: Maximum recursion depth for regular URLs (default: 3)
        max_concurrent: Maximum number of concurrent browser sessions (default: 10)
        chunk_size: Maximum size of each content chunk in characters (default: 1000)


    Returns:
        JSON string with crawl summary and storage information
    """
    logging.info(f"Invoked smart_crawl_url with url: {url}, max_depth: {max_depth}, max_concurrent: {max_concurrent}, chunk_size: {chunk_size}")

    try:
        # Get the crawler and Supabase client from the context
        crawler = ctx.request_context.lifespan_context.crawler
        supabase_client = ctx.request_context.lifespan_context.supabase_client

        crawl_results = []
        crawl_type = "webpage"

        # Detect URL type and use appropriate crawl method
        if is_txt(url):
            # For text files, use simple crawl
            crawl_results = await crawl_markdown_file(crawler, url)
            crawl_type = "text_file"
        elif is_sitemap(url):
            # For sitemaps, extract URLs and crawl in parallel
            sitemap_urls = parse_sitemap(url)
            if not sitemap_urls:
                return json.dumps({
                    "success": False,
                    "url": url,
                    "error": "No URLs found in sitemap"
                }, indent=2)
            crawl_results = await crawl_batch(crawler, sitemap_urls, max_concurrent=max_concurrent)
            crawl_type = "sitemap"
        else:
            # For regular URLs, use recursive crawl
            crawl_results = await crawl_recursive_internal_links(crawler, [url], max_depth=max_depth, max_concurrent=max_concurrent)
            crawl_type = "webpage"

        if not crawl_results:
            return json.dumps({
                "success": False,
                "url": url,
                "error": "No content found"
            }, indent=2)

        # Process results and store in Supabase
        urls = []
        chunk_numbers = []
        contents = []
        metadatas = []
        chunk_count = 0

        for doc in crawl_results:
            source_url = doc['url']
            md = doc['markdown']
            chunks = smart_chunk_markdown(md, chunk_size=chunk_size)

            for i, chunk in enumerate(chunks):
                urls.append(source_url)
                chunk_numbers.append(i)
                contents.append(chunk)

                # Extract metadata
                meta = extract_section_info(chunk)
                meta["chunk_index"] = i
                meta["url"] = source_url
                meta["source"] = urlparse(source_url).netloc
                meta["crawl_type"] = crawl_type
                meta["crawl_time"] = str(
                    asyncio.current_task().get_coro().__name__)
                metadatas.append(meta)

                chunk_count += 1

        # Create url_to_full_document mapping
        url_to_full_document = {}
        for doc in crawl_results:
            url_to_full_document[doc['url']] = doc['markdown']

        # Add to Supabase
        # IMPORTANT: Adjust this batch size for more speed if you want! Just don't overwhelm your system or the embedding API ;)
        batch_size = 20
        add_documents_to_supabase(supabase_client, urls, chunk_numbers,
                                  contents, metadatas, url_to_full_document, batch_size=batch_size)

        return json.dumps({
            "success": True,
            "url": url,
            "crawl_type": crawl_type,
            "pages_crawled": len(crawl_results),
            "chunks_stored": chunk_count,
            "urls_crawled": [doc['url'] for doc in crawl_results][:5] + (["..."] if len(crawl_results) > 5 else [])
        }, indent=2)
    except Exception as e:
        logging.error(f"Error in smart_crawl_url: {e}")
        return json.dumps({
            "success": False,
            "url": url,
            "error": str(e)
        }, indent=2)


async def crawl_markdown_file(crawler: AsyncWebCrawler, url: str) -> List[Dict[str, Any]]:
    """
    Crawl a .txt or markdown file.


    Args:
        crawler: AsyncWebCrawler instance
        url: URL of the file


    Returns:
        List of dictionaries with URL and markdown content
    """
    crawl_config = CrawlerRunConfig(
        page_timeout=60000,
        wait_until="networkidle"
    )

    try:
        result = await asyncio.wait_for(
            crawler.arun(url=url, config=crawl_config),
            timeout=90.0
        )
        if result.success and result.markdown:
            logger.info(f"Successfully crawled markdown file: {url}")
            return [{'url': url, 'markdown': result.markdown}]
        else:
            logger.warning(f"Failed to crawl {url}: {result.error_message}")
            return []
    except asyncio.TimeoutError:
        logger.error(f"Timeout crawling markdown file {url}")
        return []
    except Exception as e:
        logger.error(f"Error crawling markdown file {url}: {e}", exc_info=True)
        return []


async def crawl_batch(crawler: AsyncWebCrawler, urls: List[str], max_concurrent: int = 10) -> List[Dict[str, Any]]:
    """
    Batch crawl multiple URLs in parallel.


    Args:
        crawler: AsyncWebCrawler instance
        urls: List of URLs to crawl
        max_concurrent: Maximum number of concurrent browser sessions


    Returns:
        List of dictionaries with URL and markdown content
    """
    crawl_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        stream=False,
        page_timeout=60000,
        wait_until="networkidle"
    )
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0,
        check_interval=1.0,
        max_session_permit=max_concurrent
    )

    try:
        logger.info(f"Starting batch crawl of {len(urls)} URLs with max_concurrent={max_concurrent}")
        results = await asyncio.wait_for(
            crawler.arun_many(urls=urls, config=crawl_config, dispatcher=dispatcher),
            timeout=300.0  # 5 minute timeout for batch operations
        )
        successful = [{'url': r.url, 'markdown': r.markdown} for r in results if r.success and r.markdown]
        logger.info(f"Batch crawl completed: {len(successful)}/{len(urls)} successful")
        return successful
    except asyncio.TimeoutError:
        logger.error(f"Batch crawl timed out after 300 seconds")
        return []
    except Exception as e:
        logger.error(f"Error in batch crawl: {e}", exc_info=True)
        return []


async def crawl_recursive_internal_links(crawler: AsyncWebCrawler, start_urls: List[str], max_depth: int = 3, max_concurrent: int = 10) -> List[Dict[str, Any]]:
    """
    Recursively crawl internal links from start URLs up to a maximum depth.


    Args:
        crawler: AsyncWebCrawler instance
        start_urls: List of starting URLs
        max_depth: Maximum recursion depth
        max_concurrent: Maximum number of concurrent browser sessions


    Returns:
        List of dictionaries with URL and markdown content
    """
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        stream=False,
        page_timeout=60000,
        wait_until="networkidle"
    )
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0,
        check_interval=1.0,
        max_session_permit=max_concurrent
    )

    visited = set()

    def normalize_url(url):
        return urldefrag(url)[0]

    current_urls = set([normalize_url(u) for u in start_urls])
    results_all = []

    for depth in range(max_depth):
        urls_to_crawl = [normalize_url(
            url) for url in current_urls if normalize_url(url) not in visited]
        urls_to_crawl = [normalize_url(
            url) for url in current_urls if normalize_url(url) not in visited]
        if not urls_to_crawl:
            break

        results = await crawler.arun_many(urls=urls_to_crawl, config=run_config, dispatcher=dispatcher)
        next_level_urls = set()

        for result in results:
            norm_url = normalize_url(result.url)
            visited.add(norm_url)

            if result.success and result.markdown:
                results_all.append(
                    {'url': result.url, 'markdown': result.markdown})
                for link in result.links.get("internal", []):
                    next_url = normalize_url(link["href"])
                    if next_url not in visited:
                        next_level_urls.add(next_url)

        current_urls = next_level_urls

    return results_all


@mcp.tool()
async def get_available_sources(ctx: Context) -> str:
    """
    Get all available sources based on unique source metadata values.

    This tool returns a list of all unique sources (domains) that have been crawled and stored
    in the database. This is useful for discovering what content is available for querying.

    Args:
        ctx: The MCP server provided context

    Returns:
        JSON string with the list of available sources
    """
    try:
        # Get the Supabase client from the context
        supabase_client = ctx.request_context.lifespan_context.supabase_client

        # Use a direct query with the Supabase client
        # This could be more efficient with a direct Postgres query but
        # I don't want to require users to set a DB_URL environment variable as well
        result = supabase_client.from_('crawled_pages')\
            .select('metadata')\
            .not_.is_('metadata->>source', 'null')\
            .execute()

        # Use a set to efficiently track unique sources
        unique_sources = set()

        # Extract the source values from the result using a set for uniqueness
        if result.data:
            for item in result.data:
                source = item.get('metadata', {}).get('source')
                if source:
                    unique_sources.add(source)

        # Convert set to sorted list for consistent output
        sources = sorted(list(unique_sources))

        return json.dumps({
            "success": True,
            "sources": sources,
            "count": len(sources)
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, indent=2)


@mcp.tool()
async def perform_rag_query(ctx: Context, query: str, source: str = None, match_count: int = 5) -> str:
    """
    Perform a RAG (Retrieval Augmented Generation) query on the stored content.

    This tool searches the vector database for content relevant to the query and returns
    the matching documents. Optionally filter by source domain.

    Use the tool to get source domains if the user is asking to use a specific tool or framework.

    Args:
        ctx: The MCP server provided context
        query: The search query
        source: Optional source domain to filter results (e.g., 'example.com')
        match_count: Maximum number of results to return (default: 5)

    Returns:
        JSON string with the search results
    """
    try:
        # Get the Supabase client from the context
        supabase_client = ctx.request_context.lifespan_context.supabase_client

        # Prepare filter if source is provided and not empty
        filter_metadata = None
        if source and source.strip():
            filter_metadata = {"source": source}

        # Perform the search
        results = search_documents(
            client=supabase_client,
            query=query,
            match_count=match_count,
            filter_metadata=filter_metadata
        )

        # Format the results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "url": result.get("url"),
                "content": result.get("content"),
                "metadata": result.get("metadata"),
                "similarity": result.get("similarity")
            })

        return json.dumps({
            "success": True,
            "query": query,
            "source_filter": source,
            "results": formatted_results,
            "count": len(formatted_results)
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "query": query,
            "error": str(e)
        }, indent=2)


@mcp.tool()
async def scan_github_repo(ctx: Context, repo_owner: str, repo_name: str) -> str:
    """
    Scan a GitHub repository for markdown files and store their content in Supabase.


    Args:
        ctx: The MCP server provided context
        repo_owner: GitHub repository owner
        repo_name: GitHub repository name


    Returns:
        JSON string with the scan results
    """
    try:
        # Get the crawler and Supabase client from the context
        crawler = ctx.request_context.lifespan_context.crawler
        supabase_client = ctx.request_context.lifespan_context.supabase_client

        # Get markdown file URLs from the GitHub repository
        docs_urls = await get_docs_urls_from_github(repo_owner, repo_name)

        if not docs_urls:
            return json.dumps({
                "success": False,
                "repo_owner": repo_owner,
                "repo_name": repo_name,
                "error": "No markdown files found in the repository"
            }, indent=2)

        # Process results and store in Supabase
        urls = []
        chunk_numbers = []
        contents = []
        metadatas = []
        chunk_count = 0

        for doc in docs_urls:
            raw_url = doc['raw_url']
            github_url = doc['github_url']

            # Crawl the markdown file
            result = await crawler.arun(url=raw_url, config=CrawlerRunConfig())

            if result.success and result.markdown:
                # Chunk the content
                chunks = smart_chunk_markdown(result.markdown)

                for i, chunk in enumerate(chunks):
                    urls.append(github_url)
                    chunk_numbers.append(i)
                    contents.append(chunk)

                    # Extract metadata
                    meta = extract_section_info(chunk)
                    meta["chunk_index"] = i
                    meta["url"] = github_url
                    meta["source"] = urlparse(github_url).netloc
                    meta["crawl_type"] = "github"
                    meta["crawl_time"] = str(
                        asyncio.current_task().get_coro().__name__)
                    meta["crawl_time"] = str(
                        asyncio.current_task().get_coro().__name__)
                    metadatas.append(meta)

                    chunk_count += 1

        # Create url_to_full_document mapping
        url_to_full_document = {}
        for doc in docs_urls:
            url_to_full_document[doc['github_url']] = result.markdown

        # Add to Supabase
        # IMPORTANT: Adjust this batch size for more speed if you want! Just don't overwhelm your system or the embedding API ;)
        batch_size = 20
        add_documents_to_supabase(supabase_client, urls, chunk_numbers,
                                  contents, metadatas, url_to_full_document, batch_size=batch_size)

        return json.dumps({
            "success": True,
            "repo_owner": repo_owner,
            "repo_name": repo_name,
            "markdown_files": len(docs_urls),
            "chunks_stored": chunk_count,
            "urls_crawled": [doc['github_url'] for doc in docs_urls][:5] + (["..."] if len(docs_urls) > 5 else [])
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "repo_owner": repo_owner,
            "repo_name": repo_name,
            "error": str(e)
        }, indent=2)


@mcp.tool()
async def scan_github_source_code(ctx: Context, repo_owner: str, repo_name: str) -> str:
    """
    Scan a GitHub repository for source code files and store their content in Supabase.


    This tool extracts source code files from a GitHub repository and stores them in Supabase
    for later retrieval and querying. This provides valuable context for LLMs by including
    implementation details, API usage examples, and architectural patterns.


    Args:
        ctx: The MCP server provided context
        repo_owner: GitHub repository owner
        repo_name: GitHub repository name


    Returns:
        JSON string with the scan results
    """
    try:
        # Get the Supabase client from the context
        supabase_client = ctx.request_context.lifespan_context.supabase_client

        # Get source code file URLs from the GitHub repository
        source_code_urls = await get_source_code_urls_from_github(repo_owner, repo_name)

        if not source_code_urls:
            return json.dumps({
                "success": False,
                "repo_owner": repo_owner,
                "repo_name": repo_name,
                "error": "No source code files found in the repository"
            }, indent=2)

        # Process results and store in Supabase
        urls = []
        chunk_numbers = []
        contents = []
        metadatas = []
        chunk_count = 0

        for doc in source_code_urls:
            raw_url = doc['raw_url']
            github_url = doc['github_url']

            # Fetch the source code file directly (no crawling needed)
            try:
                response = requests.get(raw_url)
                response.raise_for_status()
                source_code = response.text
            except Exception as e:
                print(f"Error fetching source code from {raw_url}: {e}")
                continue

            if source_code:
                # Chunk the content
                chunks = smart_chunk_markdown(source_code)

                for i, chunk in enumerate(chunks):
                    urls.append(github_url)
                    chunk_numbers.append(i)
                    contents.append(chunk)

                    # Extract metadata
                    meta = extract_section_info(chunk)
                    meta["chunk_index"] = i
                    meta["url"] = github_url
                    meta["source"] = urlparse(github_url).netloc
                    meta["crawl_type"] = "github_source_code"
                    meta["file_extension"] = os.path.splitext(urlparse(github_url).path)[1]
                    meta["crawl_time"] = str(
                        asyncio.current_task().get_coro().__name__)
                    metadatas.append(meta)

                    chunk_count += 1

        # Create url_to_full_document mapping
        url_to_full_document = {}
        for doc in source_code_urls:
            raw_url = doc['raw_url']
            try:
                response = requests.get(raw_url)
                response.raise_for_status()
                url_to_full_document[doc['github_url']] = response.text
            except Exception as e:
                print(f"Error fetching source code from {raw_url}: {e}")
                url_to_full_document[doc['github_url']] = ""

        # Add to Supabase
        # IMPORTANT: Adjust this batch size for more speed if you want! Just don't overwhelm your system or the embedding API ;)
        batch_size = 20
        add_documents_to_supabase(supabase_client, urls, chunk_numbers,
                                  contents, metadatas, url_to_full_document, batch_size=batch_size)

        return json.dumps({
            "success": True,
            "repo_owner": repo_owner,
            "repo_name": repo_name,
            "source_code_files": len(source_code_urls),
            "chunks_stored": chunk_count,
            "urls_crawled": [doc['github_url'] for doc in source_code_urls][:5] + (["..."] if len(source_code_urls) > 5 else [])
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "repo_owner": repo_owner,
            "repo_name": repo_name,
            "error": str(e)
        }, indent=2)


async def get_docs_urls_from_github(repo_owner: str, repo_name: str) -> List[str]:
    """
    Pulls doc URLs from a GitHub repository.
    """
    # Set up API credentials (if you have them)
    api_token = os.getenv("GITHUB_TOKEN")
    headers = {"Accept": "application/vnd.github.v3+json"}

    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"
    else:
        print("No GitHub token provided. Fetching without authentication.")

    tree_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/git/trees/main?recursive=1"

    try:
        response = requests.get(tree_url, headers=headers)
        response.raise_for_status()
        data = response.json()

        docs_urls = []

        # Check if we have a tree in the response
        if "tree" not in data:
            print(f"No tree found in response: {data}")
            return docs_urls

        # Find markdown files in the repository
        for item in data["tree"]:
            if item["type"] == "blob" and item["path"].endswith(".md"):
                # Use the raw content URL for direct access to the file
                raw_url = f"https://raw.githubusercontent.com/{repo_owner}/{repo_name}/main/{item['path']}"
                # Use the GitHub UI URL for displaying in the results
                github_url = f"https://github.com/{repo_owner}/{repo_name}/blob/main/{item['path']}"

                docs_urls.append(
                    {"raw_url": raw_url, "github_url": github_url})
                print(f"Found markdown file: {github_url}")

        return docs_urls

    except Exception as e:
        print(f"Error fetching GitHub tree or file: {str(e)}")
        return []


async def get_source_code_urls_from_github(repo_owner: str, repo_name: str) -> List[str]:
    """
    Pulls source code URLs from a GitHub repository.
    """
    # Set up API credentials (if you have them)
    api_token = os.getenv("GITHUB_TOKEN")
    headers = {"Accept": "application/vnd.github.v3+json"}

    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"
    else:
        print("No GitHub token provided. Fetching without authentication.")

    tree_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/git/trees/main?recursive=1"

    try:
        response = requests.get(tree_url, headers=headers)
        response.raise_for_status()
        data = response.json()

        source_code_urls = []

        # Check if we have a tree in the response
        if "tree" not in data:
            print(f"No tree found in response: {data}")
            return source_code_urls

        # Common source code file extensions
        source_extensions = {
            '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h',
            '.cs', '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.scala',
            '.sql', '.sh', '.pl', '.pm', '.r', '.m', '.mm', '.dart', '.lua',
            '.groovy', '.clj', '.cljs', '.ex', '.exs', '.erl', '.hrl', '.fs',
            '.fsx', '.ml', '.mli', '.hs', '.lhs', '.coffee', '.elm', '.jl',
            '.nim', '.cr', '.v', '.zig', '.f', '.f90', '.f95', '.ada', '.adb',
            '.ads', '.pas', '.d', '.vala', '.purs', '.idr', '.agda', '.lean'
        }

        # Find source code files in the repository
        for item in data["tree"]:
            if item["type"] == "blob":
                # Check if the file has a source code extension
                _, ext = os.path.splitext(item["path"])
                if ext.lower() in source_extensions:
                    # Use the raw content URL for direct access to the file
                    raw_url = f"https://raw.githubusercontent.com/{repo_owner}/{repo_name}/main/{item['path']}"
                    # Use the GitHub UI URL for displaying in the results
                    github_url = f"https://github.com/{repo_owner}/{repo_name}/blob/main/{item['path']}"

                    source_code_urls.append(
                        {"raw_url": raw_url, "github_url": github_url})
                    print(f"Found source code file: {github_url}")

        return source_code_urls

    except Exception as e:
        print(f"Error fetching GitHub tree or file: {str(e)}")
        return []


# Define request/response models


class CrawlRequest(BaseModel):
    url: str


class SearchRequest(BaseModel):
    query: str
    match_count: int = 5


class GitHubScanRequest(BaseModel):
    repo_owner: str
    repo_name: str

@app.post("/invoke_tool")
async def invoke_tool(tool_name: str, params: Dict[str, Any]):
    """
    Generic tool invocation endpoint with proper context management.
    """
    try:
        logger.info(f"Invoking tool: {tool_name} with params: {params}")

        # Create a mock context for HTTP endpoint calls
        # This ensures the MCP tools can access the lifespan context
        class MockContext:
            def __init__(self, lifespan_ctx):
                self.request_context = type('obj', (object,), {
                    'lifespan_context': lifespan_ctx
                })()

        # Get the lifespan context from the MCP server
        # Note: This assumes the MCP server's lifespan has been initialized
        async with mcp._lifespan_manager() as lifespan_ctx:
            mock_ctx = MockContext(lifespan_ctx)

            # Map tool names to their corresponding functions
            if tool_name == "crawl_single_page":
                result = await crawl_single_page(mock_ctx, **params)
            elif tool_name == "smart_crawl_url":
                result = await smart_crawl_url(mock_ctx, **params)
            elif tool_name == "get_available_sources":
                result = await get_available_sources(mock_ctx)
            elif tool_name == "perform_rag_query":
                result = await perform_rag_query(mock_ctx, **params)
            elif tool_name == "scan_github_repo":
                result = await scan_github_repo(mock_ctx, **params)
            elif tool_name == "scan_github_source_code":
                result = await scan_github_source_code(mock_ctx, **params)
            else:
                raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")

            return {"result": result}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error invoking tool {tool_name}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/crawl")
async def crawl_endpoint(request: CrawlRequest):
    """Crawl a URL and store in Supabase."""
    try:
        logger.info(f"Crawl endpoint called for URL: {request.url}")
        result = await invoke_tool("smart_crawl_url", {"url": request.url})
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in crawl endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
async def search_endpoint(request: SearchRequest):
    """Search stored documents using RAG."""
    try:
        logger.info(f"Search endpoint called with query: {request.query}")
        result = await invoke_tool("perform_rag_query", {
            "query": request.query,
            "match_count": request.match_count
        })
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in search endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/scan_github")
async def scan_github_endpoint(request: GitHubScanRequest):
    """Scan a GitHub repository for markdown documentation."""
    try:
        logger.info(f"GitHub scan endpoint called for {request.repo_owner}/{request.repo_name}")
        result = await invoke_tool("scan_github_repo", {
            "repo_owner": request.repo_owner,
            "repo_name": request.repo_name
        })
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in GitHub scan endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/scan_github_source_code")
async def scan_github_source_code_endpoint(request: GitHubScanRequest):
    """Scan a GitHub repository for source code."""
    try:
        logger.info(f"GitHub source code scan endpoint called for {request.repo_owner}/{request.repo_name}")
        result = await invoke_tool("scan_github_source_code", {
            "repo_owner": request.repo_owner,
            "repo_name": request.repo_name
        })
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in GitHub source code scan endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """
    Health check endpoint to validate all system dependencies.
    """
    from utils import validate_ollama_connection, validate_supabase_connection

    health_status = {
        "status": "healthy",
        "timestamp": asyncio.get_event_loop().time(),
        "components": {}
    }

    # Check Supabase connection
    try:
        supabase_client = get_supabase_client()
        supabase_ok = validate_supabase_connection(supabase_client)
        health_status["components"]["supabase"] = {
            "status": "healthy" if supabase_ok else "unhealthy",
            "url": os.getenv("SUPABASE_URL", "not_set")
        }
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["components"]["supabase"] = {
            "status": "unhealthy",
            "error": str(e)
        }

    # Check Ollama connection
    ollama_ok = validate_ollama_connection()
    health_status["components"]["ollama"] = {
        "status": "healthy" if ollama_ok else "degraded",
        "url": os.getenv("OLLAMA_BASE_URL", "not_set"),
        "model": os.getenv("OLLAMA_MODEL", "not_set"),
        "note": "Will use fallback embeddings if unavailable"
    }

    # Check crawler status (basic check)
    health_status["components"]["crawler"] = {
        "status": "healthy",
        "browser": "chromium"
    }

    # Set overall status based on critical components
    if health_status["components"]["supabase"]["status"] == "unhealthy":
        health_status["status"] = "unhealthy"
    elif health_status["components"]["ollama"]["status"] == "degraded":
        health_status["status"] = "degraded"

    status_code = 200 if health_status["status"] in ["healthy", "degraded"] else 503

    from fastapi.responses import JSONResponse
    return JSONResponse(content=health_status, status_code=status_code)


@app.get("/openapi.json")
async def get_openapi():
    openapi_spec = app.openapi()
    return openapi_spec

# Update main function to use FastAPI


async def run_health_server():
    """Run a simple health check server alongside the MCP server."""
    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8051")),
        log_level="warning"  # Suppress verbose logs
    )
    server = uvicorn.Server(config)
    await server.serve()


async def main():
    transport = os.getenv("TRANSPORT", "sse")
    if transport == 'sse':
        # Run the MCP server with sse transport
        # Note: FastMCP already runs its own uvicorn server which includes our app routes
        logger.info(f"Starting MCP server with SSE transport on {mcp.settings.host}:{mcp.settings.port}")
        await mcp.run_sse_async()
    else:
        # Run standalone FastAPI server
        logger.info("Starting standalone FastAPI server")
        await run_health_server()

if __name__ == "__main__":
    asyncio.run(main())
