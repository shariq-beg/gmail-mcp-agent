# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
# This section imports async, JSON, path, and MCP client helpers used by the smoke test.

import asyncio
import json
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient


# -----------------------------------------------------------------------------
# Test Configuration
# -----------------------------------------------------------------------------
# This section defines the Gmail query, safety flag, and MCP server path used by the test.

QUERY = "category:promotions"
MAX_RESULTS = 5
ENABLE_TRASH_TEST = False
MCP_SERVER_SCRIPT = Path(__file__).with_name("server.py")


# -----------------------------------------------------------------------------
# Test Helpers
# -----------------------------------------------------------------------------
# This section contains helper functions for printing JSON and loading Gmail MCP tools.

def dump(value) -> None:
    """Print a value as pretty JSON for smoke-test output.
    Args:
        value: JSON-serializable value to display.
    Returns:
        None.
    Side effects:
        Writes formatted JSON to stdout.
    Used by:
        main in this smoke-test script.
    """
    print(json.dumps(value, indent=2, ensure_ascii=True))


async def load_gmail_tools():
    """Load Gmail MCP tools from the local MCP server process.
    Args:
        None.
    Returns:
        A dictionary mapping MCP tool names to tool objects.
    Side effects:
        Starts or connects to the local MCP server over stdio.
    Used by:
        main in this smoke-test script.
    """
    client = MultiServerMCPClient(
        {
            "gmail": {
                "command": "python",
                "args": [str(MCP_SERVER_SCRIPT)],
                "transport": "stdio",
            }
        }
    )
    tools = await client.get_tools(server_name="gmail")
    return {tool.name: tool for tool in tools}


# -----------------------------------------------------------------------------
# Smoke Test Entrypoint
# -----------------------------------------------------------------------------
# This section runs the Gmail MCP smoke-test flow when the file is executed.

async def main() -> None:
    """Run a non-destructive Gmail MCP smoke-test flow.
    Args:
        None.
    Returns:
        None.
    Side effects:
        Prints test progress and may move a message to Trash if ENABLE_TRASH_TEST is True.
    Used by:
        command-line execution of this module.
    """
    print("1. Loading Gmail MCP tools")
    tools = await load_gmail_tools()
    print(f"   Loaded tools: {', '.join(sorted(tools))}")

    print("\n2. Testing search_emails()")
    search_result = await tools["search_gmail"].ainvoke(
        {"query": QUERY, "max_results": MAX_RESULTS}
    )
    results = search_result.get("messages", [])
    print(f"   Found {len(results)} message refs")
    dump(results)

    if not results:
        print("\nNo matching emails found. Skipping read, preview, and trash tests.")
        return

    first_id = results[0]["id"]

    print("\n3. Testing read_email()")
    email_data = await tools["get_gmail_message"].ainvoke({"message_id": first_id})
    print("   Email preview:")
    dump(email_data)

    print("\n4. Testing preview_delete()")
    preview_result = await tools["preview_delete"].ainvoke(
        {"query": QUERY, "max_results": min(3, MAX_RESULTS)}
    )
    candidates = preview_result.get("candidates", [])
    print(f"   Generated {len(candidates)} delete previews")
    dump(candidates)

    print("\n5. Testing move_gmail_message_to_trash()")
    if ENABLE_TRASH_TEST:
        trash_result = await tools["move_gmail_message_to_trash"].ainvoke(
            {"message_id": first_id}
        )
        print("   Trash result:")
        dump(trash_result)
    else:
        print("   Skipped. Set ENABLE_TRASH_TEST = True to move the first matched email to Trash.")


if __name__ == "__main__":
    asyncio.run(main())
