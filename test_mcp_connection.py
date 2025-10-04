#!/usr/bin/env python3
"""
Test script to verify MCP server connection and list available tools.
"""
import requests
import json
import sys

MCP_SERVER_URL = "http://192.168.1.10:8051"

def test_sse_endpoint():
    """Test if SSE endpoint is accessible."""
    print("Testing SSE endpoint...")
    try:
        response = requests.get(f"{MCP_SERVER_URL}/sse", timeout=5, stream=True)
        print(f"✓ SSE endpoint accessible: {response.status_code}")

        # Read first few bytes
        for i, chunk in enumerate(response.iter_content(chunk_size=100)):
            if i >= 3:
                break
            print(f"  Received chunk {i}: {chunk[:100]}")

        return True
    except requests.exceptions.Timeout:
        print("✗ SSE endpoint timeout")
        return False
    except Exception as e:
        print(f"✗ SSE endpoint error: {e}")
        return False


def get_session_endpoint():
    """Get the session endpoint from SSE."""
    try:
        response = requests.get(f"{MCP_SERVER_URL}/sse", timeout=5, stream=True)

        # Read the first SSE event which contains the endpoint
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    endpoint = line[6:].strip()  # Remove 'data: ' prefix
                    return endpoint

        return None
    except Exception as e:
        print(f"Error getting session endpoint: {e}")
        return None


def test_mcp_initialize():
    """Test MCP initialization handshake."""
    print("\nTesting MCP initialization...")

    # Get the session endpoint
    session_endpoint = get_session_endpoint()
    if not session_endpoint:
        print("✗ Could not get session endpoint")
        return False

    print(f"  Using session endpoint: {session_endpoint}")

    # MCP initialize message
    init_message = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "test-client",
                "version": "1.0.0"
            }
        }
    }

    try:
        # POST to the session endpoint
        response = requests.post(
            f"{MCP_SERVER_URL}{session_endpoint}",
            json=init_message,
            headers={"Content-Type": "application/json"},
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()
            print(f"✓ Initialize successful")
            print(f"  Server info: {result.get('result', {}).get('serverInfo', {})}")
            return True, session_endpoint
        else:
            print(f"✗ Initialize failed: {response.status_code}")
            print(f"  Response: {response.text}")
            return False, None

    except Exception as e:
        print(f"✗ Initialize error: {e}")
        return False, None


def test_list_tools(session_endpoint):
    """Test listing available MCP tools."""
    print("\nTesting tools/list...")

    if not session_endpoint:
        print("✗ No session endpoint available")
        return False

    list_message = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {}
    }

    try:
        response = requests.post(
            f"{MCP_SERVER_URL}{session_endpoint}",
            json=list_message,
            headers={"Content-Type": "application/json"},
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()
            tools = result.get('result', {}).get('tools', [])
            print(f"✓ Found {len(tools)} tools:")
            for tool in tools:
                print(f"  - {tool.get('name')}: {tool.get('description', '')[:60]}...")
            return True
        else:
            print(f"✗ List tools failed: {response.status_code}")
            print(f"  Response: {response.text}")
            return False

    except Exception as e:
        print(f"✗ List tools error: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("MCP Server Connection Test")
    print(f"Server URL: {MCP_SERVER_URL}")
    print("=" * 60)

    results = []

    # Test 1: SSE endpoint
    results.append(("SSE Endpoint", test_sse_endpoint()))

    # Test 2: Initialize
    init_result, session_endpoint = test_mcp_initialize()
    results.append(("MCP Initialize", init_result))

    # Test 3: List tools (needs session from initialize)
    if init_result and session_endpoint:
        results.append(("List Tools", test_list_tools(session_endpoint)))
    else:
        results.append(("List Tools", False))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("=" * 60)

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(result for _, result in results)

    if all_passed:
        print("\n✓ All tests passed! MCP server is working correctly.")
        return 0
    else:
        print("\n✗ Some tests failed. Check the logs above for details.")
        print("\nTroubleshooting:")
        print("1. Verify the server is running: docker-compose ps")
        print("2. Check server logs: docker-compose logs mcp-server")
        print("3. Verify network connectivity: ping 192.168.1.10")
        print("4. Test basic HTTP: curl http://192.168.1.10:8051/sse")
        return 1


if __name__ == "__main__":
    sys.exit(main())
