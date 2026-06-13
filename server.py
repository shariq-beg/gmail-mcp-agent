# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
# This section imports FastMCP and Gmail client functions used by the MCP tools below.

from mcp.server.fastmcp import FastMCP

from gmail_client import (
    search_emails,
    search_emails_by_date_window,
    read_email,
    apply_label_to_email,
    preview_delete_candidates,
    trash_emails_by_label,
    trash_email,
)


# -----------------------------------------------------------------------------
# MCP Server Setup
# -----------------------------------------------------------------------------
# This section creates the FastMCP server instance used by the tool definitions.

mcp = FastMCP("gmail-mcp")


# -----------------------------------------------------------------------------
# Gmail MCP Tool Definitions
# -----------------------------------------------------------------------------
# This section exposes Gmail search, read, label, preview, and trash operations as MCP tools.

@mcp.tool()
def search_gmail(query: str, max_results: int = 50) -> dict:
    """Search Gmail using Gmail query syntax and return matching message refs.
    Args:
        query: Gmail search query, for example 'category:promotions older_than:30d'
        max_results: Maximum number of matching message references to return
    Returns:
        A dictionary with the query, result count, and Gmail message reference list.
    Used by:
        MCP clients and agent Gmail search flows through the FastMCP tool registry.
    """
    results = search_emails(query=query, max_results=max_results)
    return {
        "query": query,
        "count": len(results),
        "messages": results,
    }


@mcp.tool()
def search_gmail_date_window(
    base_query: str = "in:inbox",
    after: str | None = None,
    before: str | None = None,
    max_results: int = 50,
) -> dict:
    """Search Gmail within an optional date window and return matching refs.
    Args:
        base_query: Gmail search query to combine with dates, for example 'in:inbox'
        after: Optional Gmail date in YYYY/MM/DD format, for example '2026/04/01'
        before: Optional Gmail date in YYYY/MM/DD format, for example '2026/04/25'
        max_results: Maximum number of matching message references to return
    Returns:
        A dictionary with the final Gmail query, result count, and message refs.
    Used by:
        MCP clients and date-window review flows in the Gmail read agent.
    """
    result = search_emails_by_date_window(
        base_query=base_query,
        after=after,
        before=before,
        max_results=max_results,
    )
    return {
        "query": result["query"],
        "count": len(result["messages"]),
        "messages": result["messages"],
    }


@mcp.tool()
def get_gmail_message(message_id: str) -> dict:
    """Read one Gmail message by ID and return normalized message details.
    Args:
        message_id: Gmail message ID returned by search_gmail
    Returns:
        A dictionary with headers, snippet, body text, labels, and metadata.
    Used by:
        MCP clients and Gmail review flows that fetch full message details.
    """
    return read_email(message_id)


@mcp.tool()
def apply_gmail_label(
    message_id: str,
    label_name: str,
    create_if_missing: bool = True,
) -> dict:
    """Apply a Gmail label to one message, creating the label if requested.
    Args:
        message_id: Gmail message ID to label
        label_name: Gmail label name to apply
        create_if_missing: Whether to create the label if it does not exist
    Returns:
        A dictionary describing the labeled message and resulting Gmail labels.
    Side effects:
        Modifies the selected Gmail message by adding a label.
    Used by:
        MCP clients and the Gmail mutation agent after user confirmation.
    """
    return apply_label_to_email(
        message_id=message_id,
        label_name=label_name,
        create_if_missing=create_if_missing,
    )


@mcp.tool()
def trash_gmail_messages_by_label(label_name: str, max_results: int = 10) -> dict:
    """Move up to max_results Gmail messages with a label to Trash.

    This is a destructive action in the Gmail sense: messages are moved to Trash,
    not permanently deleted.
    Args:
        label_name: Gmail label name or label ID to search
        max_results: Maximum number of labeled messages to move to Trash
    Returns:
        A dictionary with label metadata, requested limit, count, and trash results.
    Side effects:
        Moves matching Gmail messages to Trash.
    Used by:
        MCP clients and confirmed Gmail mutation flows that target labeled messages.
    """
    return trash_emails_by_label(label_name=label_name, max_results=max_results)


@mcp.tool()
def preview_delete(query: str, max_results: int = 10) -> dict:
    """Preview Gmail messages matching a delete query without changing them.
    Args:
        query: Gmail search query used to find delete candidates
        max_results: Maximum number of preview candidates to return
    Returns:
        A dictionary with the query, candidate count, and lightweight previews.
    Used by:
        MCP clients and safety flows that inspect messages before trash actions.
    """
    previews = preview_delete_candidates(query=query, max_results=max_results)
    return {
        "query": query,
        "count": len(previews),
        "candidates": previews,
    }


@mcp.tool()
def move_gmail_message_to_trash(message_id: str) -> dict:
    """Move one Gmail message to Trash by message ID.
    Args:
        message_id: Gmail message ID to move to Trash
    Returns:
        A dictionary describing the trashed message and resulting Gmail labels.
    Side effects:
        Moves the selected Gmail message to Trash.
    Used by:
        MCP clients and the Gmail mutation agent after user confirmation.
    """
    return trash_email(message_id)


# -----------------------------------------------------------------------------
# CLI Entrypoint
# -----------------------------------------------------------------------------
# This section starts the MCP server when this module is run directly.

if __name__ == "__main__":
    mcp.run()
