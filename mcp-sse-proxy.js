#!/usr/bin/env node

/**
 * MCP SSE Proxy - Bridges SSE transport to STDIO for Claude Desktop
 * Usage: node mcp-sse-proxy.js <sse-url>
 */

const fetch = require('node:fetch');
const readline = require('node:readline');

const SSE_URL = process.argv[2];
if (!SSE_URL) {
  console.error('Usage: mcp-sse-proxy.js <sse-url>');
  process.exit(1);
}

let sessionUrl = null;

async function connectToSSE() {
  try {
    // Get session endpoint
    const response = await fetch(SSE_URL);
    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    let buffer = '';
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          sessionUrl = line.slice(6).trim();
          console.error(`Session URL: ${sessionUrl}`);
          break;
        }
      }
      if (sessionUrl) break;
    }

    if (!sessionUrl) {
      throw new Error('Failed to get session URL from SSE endpoint');
    }

    // Now handle MCP protocol over session
    startMCPProxy();

  } catch (error) {
    console.error('SSE connection error:', error.message);
    process.exit(1);
  }
}

function startMCPProxy() {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    terminal: false
  });

  rl.on('line', async (line) => {
    try {
      const request = JSON.parse(line);

      // Forward to MCP server
      const response = await fetch(sessionUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request)
      });

      const result = await response.json();
      console.log(JSON.stringify(result));

    } catch (error) {
      console.error('Proxy error:', error.message);
    }
  });
}

connectToSSE();
