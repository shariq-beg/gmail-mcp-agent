# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
# This section imports CLI, async, JSON, LangChain, and agent helpers used by diagnostics.

from __future__ import annotations

import argparse
import asyncio
import json

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.prebuilt import create_react_agent

from agent import (
    DEFAULT_MODEL,
    SYSTEM_PROMPT,
    _message_text,
    build_llm,
    load_tools,
)


# -----------------------------------------------------------------------------
# Diagnostic Checks
# -----------------------------------------------------------------------------
# This section contains smoke checks for the LLM, local tools, and Gmail MCP access.


async def smoke_test(model: str, temperature: float) -> None:
    """Run an agent smoke test against the configured LLM and local tools.
    Args:
        model: Model name passed to the OpenAI-compatible chat client.
        temperature: Sampling temperature used for the diagnostic LLM call.
    Returns:
        None.
    Side effects:
        Prints loaded tools and diagnostic responses to the terminal.
    Used by:
        main when the diagnostics CLI is run with the agent check.
    """
    tools = await load_tools()
    print("Loaded tools:")
    for item in tools:
        print(f"- {item.name}")

    llm = build_llm(model=model, temperature=temperature)
    response = await llm.ainvoke("Reply with exactly: agent smoke test ok")
    print("\nLLM response:")
    print(_message_text(response))

    graph = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)
    graph_response = await graph.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Use the list_review_sessions tool with max_results=1. "
                        "Then reply with exactly: tool smoke test ok"
                    )
                )
            ]
        }
    )
    last_ai = next(
        (
            message
            for message in reversed(graph_response["messages"])
            if isinstance(message, AIMessage)
        ),
        None,
    )
    print("\nTool-call response:")
    print(_message_text(last_ai) if last_ai else "No AI response")


async def gmail_smoke_test() -> None:
    """Run a Gmail MCP smoke test by invoking the search_gmail tool.
    Args:
        None.
    Returns:
        None.
    Side effects:
        Connects to local MCP tooling and prints a JSON Gmail search result.
    Used by:
        main when the diagnostics CLI is run with the gmail check.
    """
    tools = await load_tools()
    search = next(item for item in tools if item.name == "search_gmail")
    result = await search.ainvoke(
        {"query": "in:inbox category:primary", "max_results": 1}
    )
    print(json.dumps(result, indent=2))


# -----------------------------------------------------------------------------
# CLI Entrypoint
# -----------------------------------------------------------------------------
# This section parses diagnostic command arguments and dispatches the selected check.

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for diagnostic checks.
    Args:
        None.
    Returns:
        Parsed argparse namespace containing model, temperature, and check.
    Used by:
        main in this module.
    """
    parser = argparse.ArgumentParser(description="Diagnostics for the Gmail MCP agent.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument(
        "check",
        choices=("agent", "gmail"),
        help="Diagnostic check to run.",
    )
    return parser.parse_args()


def main() -> None:
    """Dispatch the requested diagnostics check from command-line arguments.
    Args:
        None.
    Returns:
        None.
    Side effects:
        Runs an async diagnostic check and prints results to the terminal.
    Used by:
        command-line execution of this module.
    """
    args = parse_args()
    if args.check == "gmail":
        asyncio.run(gmail_smoke_test())
    else:
        asyncio.run(smoke_test(model=args.model, temperature=args.temperature))


if __name__ == "__main__":
    main()
