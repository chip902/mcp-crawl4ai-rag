FROM python:3.12-slim

ARG PORT=8051

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the MCP server files
COPY . .

# Install packages directly to the system (no virtual environment)
# Combining commands to reduce Docker layers
RUN uv pip install --system -e . && \
    crawl4ai-setup

# Install Playwright browsers
RUN playwright install --with-deps chromium

EXPOSE ${PORT}

# Command to run the MCP server
# CMD ["uv", "run", "src/crawl4ai_mcp.py"]
CMD ["python", "src/crawl4ai_mcp.py"]
