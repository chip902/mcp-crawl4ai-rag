#!/usr/bin/env python3
"""
MCP SSE-to-STDIO Proxy for Claude Desktop
Connects to an SSE MCP server and bridges it to STDIO transport
"""

import sys
import json
import asyncio
import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SSE_URL = "http://192.168.1.10:8051/sse"


async def main():
    """Run the SSE proxy"""
    async with httpx.AsyncClient() as client:
        async with client.stream("GET", SSE_URL) as response:
            # Read SSE stream to get session endpoint
            session_url = None
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    session_url = line[6:].strip()
                    break

            if not session_url:
                print("Failed to get session URL", file=sys.stderr)
                sys.exit(1)

            print(f"Connected to session: {session_url}", file=sys.stderr)

            # Now proxy MCP messages between STDIO and the session
            async def read_stdin():
                loop = asyncio.get_event_loop()
                reader = asyncio.StreamReader()
                protocol = asyncio.StreamReaderProtocol(reader)
                await loop.connect_read_pipe(lambda: protocol, sys.stdin)
                return reader

            stdin_reader = await read_stdin()

            while True:
                try:
                    # Read from stdin
                    line = await stdin_reader.readline()
                    if not line:
                        break

                    # Parse and forward to MCP server
                    message = json.loads(line.decode())

                    # Send to session endpoint
                    resp = await client.post(
                        session_url,
                        json=message,
                        headers={"Content-Type": "application/json"}
                    )

                    # Return response to stdout
                    result = resp.json()
                    print(json.dumps(result), flush=True)

                except Exception as e:
                    print(f"Error: {e}", file=sys.stderr)
                    break


if __name__ == "__main__":
    asyncio.run(main())
