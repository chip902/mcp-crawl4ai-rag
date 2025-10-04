# MCP Crawl4AI RAG Server - Deployment Guide

## Overview
This document provides deployment instructions and troubleshooting guidance for the MCP Crawl4AI RAG server.

## Recent Improvements

### 1. Comprehensive Timeout & Retry Logic ✅
- Added 60-second page timeouts for all crawl operations
- Added 90-second overall timeout for single page crawls
- Added 5-minute timeout for batch crawl operations
- Implemented exponential backoff retry (3 attempts) for embedding generation
- All operations now fail fast with clear error messages

### 2. Docker Networking & Health Checks ✅
- Updated `docker-compose.yml` with proper environment variable injection
- Added health check endpoint at `/health` (checked every 30s)
- Configured `extra_hosts` for proper Docker networking with `host.docker.internal`
- Added `restart: unless-stopped` for automatic recovery
- Health checks validate Supabase, Ollama, and Crawler status

### 3. Ollama Connection Improvements ✅
- Added connection validation on startup
- Implemented connection caching to reduce validation overhead
- Graceful degradation: falls back to deterministic embeddings when Ollama unavailable
- Retry logic with exponential backoff for transient failures
- Clear logging when using fallback mode

### 4. Enhanced Error Handling & Logging ✅
- Replaced all `print()` statements with structured logging
- Added log levels: INFO, WARNING, ERROR, DEBUG
- Detailed error context with file:line numbers
- Request/response logging for debugging
- Stack traces for unexpected errors

### 5. Health Check Endpoint ✅
- `GET /health` returns JSON with component status
- Returns 200 (healthy), 200 (degraded), or 503 (unhealthy)
- Validates:
  - Supabase connectivity
  - Ollama availability
  - Crawler initialization

### 6. MCP Context Management ✅
- Fixed context lifecycle for HTTP endpoints
- Proper async context management
- All endpoints now share crawler/Supabase instances
- Prevented context leaks and resource exhaustion

### 7. Configuration Validation ✅
- Startup validation of required environment variables
- Clear error messages for missing configuration
- Logs sanitized configuration (hides secrets)
- Non-blocking Ollama validation (warns but doesn't fail)

## Deployment Instructions

### Prerequisites
- Docker and Docker Compose installed
- Supabase instance running at `192.168.1.10:8000`
- Ollama instance running at `192.168.1.10:11434` (optional)
- `.env` file with required variables

### Environment Variables

Required:
```env
SUPABASE_URL=http://192.168.1.10:8000
SUPABASE_SERVICE_KEY=your_service_key_here
```

Optional (with defaults):
```env
TRANSPORT=sse
HOST=0.0.0.0
PORT=8051
OLLAMA_BASE_URL=http://192.168.1.10:11434
OLLAMA_MODEL=nomic-embed-text
MODEL_CHOICE=nomic-embed-text
GITHUB_TOKEN=your_github_token
```

### Build and Deploy

1. **Build the Docker image:**
   ```bash
   docker build -t mcp/crawl4ai-rag:latest .
   ```

2. **Start with Docker Compose:**
   ```bash
   docker-compose up -d
   ```

3. **Check health:**
   ```bash
   curl http://192.168.1.10:8051/health
   ```

4. **View logs:**
   ```bash
   docker-compose logs -f mcp-server
   ```

### Connecting from LLMs/Tools

#### MCP SSE Connection (Default)
```json
{
  "mcpServers": {
    "crawl4ai-rag": {
      "url": "http://192.168.1.10:8051/sse",
      "transport": "sse"
    }
  }
}
```

#### Direct HTTP API
```bash
# Crawl a URL
curl -X POST http://192.168.1.10:8051/crawl \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com"}'

# Search documents
curl -X POST http://192.168.1.10:8051/search \
  -H "Content-Type: application/json" \
  -d '{"query": "your search query", "match_count": 5}'
```

## Troubleshooting

### Issue: Timeouts or Slow Responses

**Symptoms:** Requests timeout or take >90 seconds

**Solutions:**
1. Check Ollama connectivity:
   ```bash
   curl http://192.168.1.10:11434/api/tags
   ```
2. Check Supabase connectivity:
   ```bash
   curl http://192.168.1.10:8000/rest/v1/
   ```
3. Review server logs:
   ```bash
   docker logs mcp-crawl4ai-rag_mcp-server_1
   ```
4. Reduce `max_concurrent` parameter (default: 10)
5. Check available memory on the host

### Issue: Embedding Failures

**Symptoms:** "Using fallback embedding" warnings in logs

**Solutions:**
1. Verify Ollama is running:
   ```bash
   docker ps | grep ollama
   ```
2. Check if model is pulled:
   ```bash
   curl http://192.168.1.10:11434/api/tags | jq '.models[] | .name'
   ```
3. Pull the model if missing:
   ```bash
   docker exec -it ollama ollama pull nomic-embed-text
   ```

**Note:** Fallback embeddings work but may reduce search quality.

### Issue: Supabase Connection Errors

**Symptoms:** "Cannot connect to Supabase" errors

**Solutions:**
1. Verify Supabase is running:
   ```bash
   curl http://192.168.1.10:8000/rest/v1/
   ```
2. Check the `crawled_pages` table exists:
   ```sql
   -- Run in Supabase SQL editor
   SELECT * FROM crawled_pages LIMIT 1;
   ```
3. Verify `SUPABASE_SERVICE_KEY` has correct permissions
4. Check network connectivity from Docker container:
   ```bash
   docker exec mcp-crawl4ai-rag_mcp-server_1 curl http://192.168.1.10:8000
   ```

### Issue: Crawler Initialization Failures

**Symptoms:** "Error initializing crawler" on startup

**Solutions:**
1. Check Playwright installation:
   ```bash
   docker exec mcp-crawl4ai-rag_mcp-server_1 playwright --version
   ```
2. Rebuild Docker image with `--no-cache`:
   ```bash
   docker build --no-cache -t mcp/crawl4ai-rag:latest .
   ```
3. Check available disk space:
   ```bash
   df -h
   ```

### Issue: Inconsistent Results from Different Clients

**Symptoms:** Works in one client but not another

**Solutions:**
1. Check health endpoint from the failing client's network
2. Verify client timeout settings (should be >90 seconds)
3. Check if client supports SSE transport
4. Try HTTP POST endpoints instead of MCP SSE
5. Review client-specific logs for authentication issues

## Performance Tuning

### For High-Volume Crawling
```python
# Adjust in smart_crawl_url call
max_concurrent = 5  # Reduce from default 10
chunk_size = 3000   # Reduce from default 5000
```

### For Faster Embeddings
1. Use a faster Ollama model (e.g., `all-minilm`)
2. Increase batch size in `add_documents_to_supabase`:
   ```python
   batch_size = 50  # Increase from default 20
   ```
3. Disable contextual embeddings:
   ```env
   MODEL_CHOICE=  # Leave empty
   ```

### For Better Search Quality
1. Use a better embedding model:
   ```env
   OLLAMA_MODEL=mxbai-embed-large
   ```
2. Enable contextual embeddings:
   ```env
   MODEL_CHOICE=nomic-embed-text
   ```
3. Reduce chunk size for more granular results:
   ```python
   chunk_size = 2000
   ```

## Monitoring

### Health Check Monitoring
```bash
# Monitor health every 30 seconds
watch -n 30 'curl -s http://192.168.1.10:8051/health | jq'
```

### Log Monitoring
```bash
# Follow logs with grep filter
docker-compose logs -f | grep ERROR

# Count warnings
docker-compose logs --since 1h | grep WARNING | wc -l
```

### Resource Monitoring
```bash
# Check container stats
docker stats mcp-crawl4ai-rag_mcp-server_1

# Check disk usage
docker system df
```

## Backup and Recovery

### Database Backup
Use Supabase's built-in backup or:
```bash
pg_dump -h 192.168.1.10 -U postgres -d your_db > backup.sql
```

### Container Recovery
```bash
# Restart unhealthy container
docker-compose restart mcp-server

# Full rebuild if needed
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

## Security Notes

1. **Never commit `.env` file** - contains sensitive keys
2. **Use service role key** for `SUPABASE_SERVICE_KEY`, not anon key
3. **Restrict network access** - use firewall rules for port 8051
4. **Monitor GitHub rate limits** when using `GITHUB_TOKEN`
5. **Rotate secrets regularly**

## Support

For issues not covered here:
1. Check application logs: `docker-compose logs -f mcp-server`
2. Check health endpoint: `curl http://192.168.1.10:8051/health`
3. Review Docker network: `docker network inspect mcp-crawl4ai-rag_app_network`
4. File an issue with logs and health check output
