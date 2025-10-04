# OpenAPI Server Setup for Open-WebUI

This project now supports **two modes of operation**:

1. **MCP Server** (port 8051) - for Claude Code, Windsurf, and other MCP-compatible clients
2. **OpenAPI REST API** (port 8081) - for Open-WebUI and other OpenAPI-compatible tools

## Quick Start

### Deploy Both Servers

```bash
# Build and start both servers
docker-compose up -d --build

# Check logs
docker-compose logs -f
```

This will start:
- MCP server on `http://192.168.1.10:8051`
- OpenAPI server on `http://192.168.1.10:8082`

### Configure in Open-WebUI

1. Go to **Settings** → **External Tools**
2. Click **Add New Connection**
3. Configure:
   - **Type**: OpenAPI
   - **URL**: `http://192.168.1.10:8082`
   - **OpenAPI Spec URL**: Select `openapi.json` from dropdown
   - **Auth**: None
   - **ID**: `crawl4ai-rag`
   - **Name**: Crawl4AI RAG
   - **Description**: Web crawling and RAG query API
4. Click **Save**

## Available Endpoints

### Interactive Documentation
- **Swagger UI**: `http://192.168.1.10:8082/docs`
- **ReDoc**: `http://192.168.1.10:8082/redoc`
- **OpenAPI Spec**: `http://192.168.1.10:8082/openapi.json`

### API Endpoints

#### Query Endpoints
- `POST /query` - Perform RAG query on stored documentation
- `GET /sources` - List all available documentation sources

#### Crawling Endpoints
- `POST /crawl/single` - Crawl a single web page
- `POST /crawl/smart` - Intelligently crawl based on URL type (currently proxies to /crawl/single)

#### GitHub Endpoints (Not Yet Implemented)
- `POST /github/docs` - Scan GitHub repo for markdown files
- `POST /github/source` - Scan GitHub repo for source code

## Example Usage

### Query Documentation

```bash
curl -X POST http://192.168.1.10:8082/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "spam filtering configuration",
    "source": "docker-mailserver.github.io",
    "match_count": 5
  }'
```

### List Sources

```bash
curl http://192.168.1.10:8082/sources
```

### Crawl a Page

```bash
curl -X POST http://192.168.1.10:8082/crawl/single \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/docs"
  }'
```

## Using in Open-WebUI

Once configured, you can use the tools in Open-WebUI chat:

**Example prompts**:
- "Query the documentation for spam filtering best practices"
- "Search for Docker configuration examples"
- "Crawl this URL and add it to the knowledge base: https://..."

## Troubleshooting

### Check Server Health

```bash
curl http://192.168.1.10:8082/health
```

Should return:
```json
{"status": "healthy"}
```

### View Logs

```bash
# OpenAPI server logs
docker-compose logs -f openapi-server

# MCP server logs
docker-compose logs -f mcp-server
```

### Restart Services

```bash
# Restart OpenAPI server only
docker-compose restart openapi-server

# Restart all services
docker-compose restart
```

## Architecture

```
┌─────────────────┐
│   Open-WebUI    │
│   (OpenAPI)     │
└────────┬────────┘
         │
         │ HTTP REST
         ├─────────────────────────┐
         │                         │
         ▼                         ▼
┌─────────────────┐       ┌─────────────────┐
│ OpenAPI Server  │       │   MCP Server    │
│   Port 8082     │       │   Port 8051     │
└────────┬────────┘       └────────┬────────┘
         │                         │
         └─────────┬───────────────┘
                   │
                   ▼
         ┌─────────────────┐
         │   Supabase      │
         │  Vector Store   │
         └─────────────────┘
                   │
                   ▼
         ┌─────────────────┐
         │    Ollama       │
         │  (Embeddings)   │
         └─────────────────┘
```

## Next Steps

- [ ] Implement smart crawling logic for sitemaps and recursive crawling
- [ ] Add GitHub scanning endpoints
- [ ] Add authentication/API keys
- [ ] Add rate limiting
- [ ] Add caching layer
