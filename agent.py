# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
# This section imports Python, LangChain, MCP, OpenAI-compatible, and LangGraph dependencies.

from __future__ import annotations

import argparse
import asyncio
import ast
import json
import warnings
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.warnings import LangGraphDeprecatedSinceV10


warnings.filterwarnings("ignore", category=LangGraphDeprecatedSinceV10)


# -----------------------------------------------------------------------------
# Project Paths And Runtime Configuration
# -----------------------------------------------------------------------------
# This section defines project files, model defaults, scan limits, categories, and progress labels.

PROJECT_DIR = Path(__file__).resolve().parent
MCP_SERVER_SCRIPT = PROJECT_DIR / "server.py"
REVIEW_SESSIONS_DIR = PROJECT_DIR / "review_sessions"
REVIEW_DB_FILE = PROJECT_DIR / "gmail_review_db.json"
_GMAIL_MCP_CLIENT: MultiServerMCPClient | None = None
_GMAIL_MCP_TOOLS: dict[str, Any] | None = None

LM_STUDIO_BASE_URL = "http://127.0.0.1:1234/v1"
LM_STUDIO_API_KEY = "lm-studio"
DEFAULT_MODEL = "qwen/qwen3.5-9b"

CONTEXT_WINDOW_TOKENS = 16_384
MAX_CATEGORIZATION_BATCH = 10
DEFAULT_SCAN_LIMIT = 30
MAX_SCAN_LIMIT = 75
BODY_PREVIEW_CHARS = 1_500
MAX_TOOL_SCAN_LIMIT = 75

Category = Literal["useful", "promotional", "job_notifications", "need_further_review"]
Route = Literal["chat_agent", "gmail_read_agent", "gmail_mutation_agent"]
ProgressCallback = Callable[[str, str], None]
REVIEW_CATEGORIES = ("useful", "promotional", "job_notifications", "need_further_review")
PROGRESS_STAGE_LABELS = {
    "routing": "Routing request...",
    "loading_gmail_mcp_tools": "Loading Gmail MCP tools...",
    "gmail_read_agent_at_work": "Gmail read agent at work...",
    "gmail_mutation_agent_at_work": "Gmail mutation agent at work...",
    "searching_gmail": "Searching Gmail...",
    "reading_gmail": "Reading Gmail messages...",
    "checking_review_cache": "Checking review cache...",
    "classifying_emails": "Classifying emails...",
    "saving_review_cache": "Saving review cache...",
    "saving_review_session": "Saving review session...",
    "executing_gmail_mutation": "Applying confirmed Gmail changes...",
    "preparing_response": "Preparing response...",
    "done": "Done",
}
_PROGRESS_CALLBACK: ContextVar[ProgressCallback | None] = ContextVar(
    "progress_callback",
    default=None,
)


# -----------------------------------------------------------------------------
# Agent System Prompts
# -----------------------------------------------------------------------------
# This section stores router, final response, Gmail read, and Gmail mutation system prompts.

ROUTER_SYSTEM_PROMPT = """
/no_think
You are an intent router for a local app with specialist agents.

Return only one visible JSON object. Do not explain, do not think aloud, do not
use markdown, and do not leave the answer blank.

The JSON must have this exact shape:
{"route":"chat_agent|gmail_read_agent|gmail_mutation_agent","reason":"short reason"}

Use gmail_read_agent when the user asks to search, scan, read, classify,
categorize, summarize, inspect, or discuss Gmail emails or saved Gmail
review-session results.

Use gmail_mutation_agent when the user asks to modify Gmail, including label,
archive, trash, delete, move, mark read/unread, or clean up messages.

Use chat_agent for greetings, thanks, general conversation, explanations about
the app, and messages that do not require Gmail tools.

If the user is replying to an immediately previous Gmail mutation question with
a short confirmation or cancellation, use gmail_mutation_agent.

If no specialist is clearly needed, use chat_agent.
""".strip()

ROUTER_REPAIR_SYSTEM_PROMPT = """
/no_think
Return only one visible JSON object for routing. Do not explain, do not think
aloud, do not use markdown, and do not leave the answer blank.

Valid routes are chat_agent, gmail_read_agent, and gmail_mutation_agent.
The JSON shape is:
{"route":"chat_agent|gmail_read_agent|gmail_mutation_agent","reason":"short reason"}
""".strip()

FINAL_RESPONSE_SYSTEM_PROMPT = """
/no_think
You are the user's consistent front-facing chat agent.

Use the conversation history plus any specialist work result to answer the user.
If specialist work is present, summarize it accurately and include useful next
steps. If no specialist was used, answer normally.

Do not invent tool results. Do not expose internal graph/node details unless the
user asks about architecture.
Respond with the final answer only.
""".strip()


# -----------------------------------------------------------------------------
# Agent State Types
# -----------------------------------------------------------------------------
# This section defines dataclasses and graph state shapes passed between workflow nodes.

@dataclass
class AgentState:
    chat_messages: list[BaseMessage] = field(default_factory=list)
    gmail_messages: list[BaseMessage] = field(default_factory=list)
    gmail_mutation_messages: list[BaseMessage] = field(default_factory=list)
    latest_review_session_path: str | None = None
    latest_review_results: list[dict[str, Any]] = field(default_factory=list)
    pending_action: dict[str, Any] | None = None
    last_specialist_outputs: list[dict[str, Any]] = field(default_factory=list)
    last_route: Route | None = None


@dataclass
class AgentTurnResult:
    response: str
    state: AgentState
    route: Route
    route_reason: str


class WorkflowState(TypedDict, total=False):
    prompt: str
    agent_state: AgentState
    model: str
    temperature: float
    ui_messages: list[dict[str, str]]
    route: Route
    route_reason: str
    specialist_name: str | None
    specialist_output: str | None
    gmail_messages: list[BaseMessage]
    gmail_mutation_messages: list[BaseMessage]
    latest_review_session_path: str | None
    latest_review_results: list[dict[str, Any]]
    pending_action: dict[str, Any] | None
    response: str
    next_agent_state: AgentState


# -----------------------------------------------------------------------------
# Progress Reporting
# -----------------------------------------------------------------------------
# This section emits progress callbacks consumed by the Streamlit UI and runtime callers.

def _emit_progress(stage: str, label: str | None = None) -> None:
    """Send a progress update to the active callback, if one is registered.
    Args:
        stage: Stable progress stage key.
        label: Optional user-facing label to override the default stage label.
    Returns:
        None.
    Side effects:
        Calls the context-local progress callback when present.
    Used by:
        Gmail tool helpers, workflow nodes, and handle_user_message.
    """
    callback = _PROGRESS_CALLBACK.get()
    if callback is None:
        return
    callback(stage, label or PROGRESS_STAGE_LABELS.get(stage, stage.replace("_", " ")))


GMAIL_READ_SYSTEM_PROMPT = f"""
/no_think
You are the Gmail read specialist running on the user's machine.

Your language model is Qwen served by LM Studio. Assume a {CONTEXT_WINDOW_TOKENS}
token context window and optimize for quality over huge batches.

You have high-level local review tools for Gmail review scans. Do not invent
Gmail results.

Primary job:
- Chat naturally with the user.
- Interpret scan instructions with the LLM. Do not rely on hard-coded parsing.
- When the user asks to scan/search/review/categorize Gmail, call exactly one
  high-level review tool: review_gmail_search or review_gmail_date_window.
- Do not manually combine raw Gmail search, cache, read, and save tools for
  normal scans. The high-level review tools enforce the cache and fresh-read
  workflow safely.
- If the user asks for a fresh scan, rescan, or says not to use cache, pass
  use_cache=false to the high-level review tool.
- Categorize emails into exactly one of:
  1. useful
  2. promotional
  3. job_notifications
  4. need_further_review
- Save review results with the local review-session tools when a scan completes.
- Let the user discuss, inspect, and change categorization results.
- Never apply labels, archive, trash, delete, or otherwise modify Gmail. Gmail
  changes are handled by the Gmail mutation specialist after review context and
  confirmation.

Date handling:
- The user is in Australia. Interpret ambiguous slash dates as DD/MM/YYYY.
- Example: 08/05/2026 means 8 May 2026, not August 5 2026.
- Convert dates to Gmail format YYYY/MM/DD before calling tools.
- Gmail before dates are exclusive. For an inclusive end date, use the day after the user's requested end date.
- Example: 08/05/2026 to 09/05/2026 should call after=2026/05/08 and before=2026/05/10.
- If a date is impossible or genuinely ambiguous, ask a clarifying question before searching.
- Mention the final date range searched in the response.

Gmail query guidance:
- Primary inbox usually means: in:inbox category:primary
- Promotions usually means: in:inbox category:promotions
- If the user gives dates, use review_gmail_date_window.
- Gmail date strings must be YYYY/MM/DD.
- If the user gives only a number of emails, respect that limit.
- If no limit is given, use {DEFAULT_SCAN_LIMIT}.
- Never scan more than {MAX_SCAN_LIMIT} emails unless the user explicitly asks.

16K batching policy:
- Keep categorization batches to at most {MAX_CATEGORIZATION_BATCH} emails.
- Start with sender, subject, date, snippet, labels, and only a short body preview.
- Fetch full details only when needed.
- The high-level review tools search Gmail, inspect message IDs, reuse cached
  records when allowed, fetch only missing messages, classify new messages, save
  new cache records, and save a review session.
- Do not paste long bodies back to the user unless they ask.
- If a message is ambiguous, mark it need_further_review.

Categorization policy:
- useful: personal, financial, legal, account/security, work, medical, travel,
  receipts/warranties the user may need, time-sensitive notices, real human
  messages, or anything likely to require future action/reference.
- job_notifications: automated job alerts or recommendations, such as LinkedIn
  or SEEK job emails. Never treat these as promotional.
- promotional: marketing, coupons, newsletters, sales, product announcements,
  bulk campaigns, social promotions, or low-value automated engagement mail.
- need_further_review: ambiguous, potentially important but unclear, missing
  context, unusual sender, suspicious, or conflicting signals.

For every categorized email, keep this structure:
{{
  "message_id": "...",
  "thread_id": "...",
  "date": "...",
  "from": "...",
  "subject": "...",
  "snippet": "...",
  "category": "useful|promotional|job_notifications|need_further_review",
  "confidence": "low|medium|high",
  "reason": "short reason"
}}

Safety:
- Reading and categorizing is allowed when the user asks.
- Gmail modification is not allowed in this specialist.

When presenting scan results:
- Give a compact count summary.
- Show a short sample of each category.
- Tell the user they can ask to inspect, reclassify, label, archive, or trash
  after reviewing.
""".strip()


GMAIL_MUTATION_SYSTEM_PROMPT = """
/no_think
You are the Gmail mutation specialist.

You never search, scan, or classify Gmail. You only prepare or execute Gmail
changes after the app has existing review results.

Rules:
- If there is no review context, mutation is blocked and the user must scan
  Gmail first.
- If a mutation is requested and review context exists, prepare a clear pending
  action preview and require user confirmation.
- Do not execute Gmail changes unless the user has confirmed a stored pending
  action.
- Prefer existing reviewed categories and message IDs over guessing from chat.
""".strip()


# -----------------------------------------------------------------------------
# Review Session Path Helpers
# -----------------------------------------------------------------------------
# This section resolves and validates local review-session paths before file access.

def _session_path_from_name(name: str) -> Path:
    """Resolve a review-session file name or path to an absolute path.
    Args:
        name: Review-session file name or path.
    Returns:
        Absolute Path for the requested review-session file.
    Used by:
        _safe_session_path before reading or updating session files.
    """
    path = Path(name)
    if not path.is_absolute():
        path = REVIEW_SESSIONS_DIR / path.name
    return path.resolve()


def _safe_session_path(name: str) -> Path:
    """Resolve and validate that a session path stays inside review_sessions.
    Args:
        name: Review-session file name or path supplied by a tool caller.
    Returns:
        Absolute Path inside REVIEW_SESSIONS_DIR.
    Raises:
        ValueError: Raised when the resolved path escapes review_sessions.
    Side effects:
        Ensures the review_sessions directory exists.
    Used by:
        load_review_session and update_review_category.
    """
    REVIEW_SESSIONS_DIR.mkdir(exist_ok=True)
    path = _session_path_from_name(name)
    root = REVIEW_SESSIONS_DIR.resolve()
    if root != path and root not in path.parents:
        raise ValueError("Session path must stay inside review_sessions.")
    return path


# -----------------------------------------------------------------------------
# Review Classification Helpers
# -----------------------------------------------------------------------------
# This section normalizes categories, prepares email previews, and parses classifier JSON.

def _category_from_legacy(value: Any) -> str:
    """Normalize legacy category names to the current review categories.
    Args:
        value: Raw category or classification value.
    Returns:
        Normalized category string.
    Used by:
        Cache normalization, cache saving, stats, and session update logic.
    """
    normalized = str(value or "").strip().lower()
    aliases = {
        "job_alert": "job_notifications",
        "job_alerts": "job_notifications",
        "job_notification": "job_notifications",
        "jobs": "job_notifications",
        "promotional_not_useful": "promotional",
        "needs_further_review": "need_further_review",
        "needs_review": "need_further_review",
    }
    return aliases.get(normalized, normalized)


def _normalize_cached_review(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize a cached review row into the current result shape.
    Args:
        row: Cached review record from gmail_review_db.json.
    Returns:
        A normalized review-result dictionary marked as cache-sourced.
    Used by:
        get_cached_email_reviews and _review_message_refs.
    """
    message_id = row.get("message_id") or row.get("id")
    return {
        "message_id": message_id,
        "thread_id": row.get("thread_id", ""),
        "date": row.get("date", ""),
        "from": row.get("from", ""),
        "subject": row.get("subject", ""),
        "snippet": row.get("snippet", ""),
        "category": _category_from_legacy(row.get("category") or row.get("classification")),
        "confidence": row.get("confidence", "medium"),
        "reason": row.get("reason", "Loaded from local review cache."),
        "review_source": "cache",
        "cached_at": row.get("cached_at") or row.get("last_seen_at") or row.get("reviewed_at"),
    }


def _email_preview_for_classification(email: dict[str, Any]) -> dict[str, Any]:
    """Create a compact email payload for classifier prompts.
    Args:
        email: Normalized Gmail message dictionary.
    Returns:
        A classifier-ready dictionary with trimmed body text and key metadata.
    Used by:
        _review_message_refs before fresh classification.
    """
    body_text = (email.get("body_text") or "").strip()
    if len(body_text) > BODY_PREVIEW_CHARS:
        body_text = body_text[:BODY_PREVIEW_CHARS] + "\n[truncated]"
    return {
        "message_id": email.get("id") or email.get("message_id"),
        "thread_id": email.get("thread_id", ""),
        "date": email.get("date", ""),
        "from": email.get("from", ""),
        "subject": email.get("subject", ""),
        "snippet": email.get("snippet", ""),
        "sender_domain": email.get("sender_domain", ""),
        "gmail_labels": email.get("gmail_labels", email.get("label_ids", [])),
        "list_unsubscribe_present": email.get("list_unsubscribe_present", False),
        "body": body_text,
        "user_replied": bool(email.get("user_replied", False)),
    }


def _classification_prompt(emails: list[dict[str, Any]]) -> str:
    """Build the JSON-only classification prompt for a batch of email previews.
    Args:
        emails: Email preview dictionaries to classify.
    Returns:
        Prompt string sent to the local OpenAI-compatible chat model.
    Used by:
        _classify_emails_with_qwen.
    """
    return (
        "/no_think\n"
        "Classify each Gmail email into exactly one category: useful, promotional, "
        "job_notifications, need_further_review.\n\n"
        "Return only one visible JSON object. Do not explain, do not think aloud, "
        "do not use markdown, and do not leave the answer blank.\n\n"
        "Definitions:\n"
        "- useful: replied emails, personal/work communication, finance/legal/medical, "
        "security alerts, receipts, travel, job applications/interviews. Keep if any future value.\n"
        "- job_notifications: automated job alerts or recommendations (e.g., LinkedIn, SEEK). Never promotional.\n"
        "- promotional: sales, newsletters, social media, product suggestions with no action/value.\n"
        "- need_further_review: suspicious, conflicting signals, or unsure.\n\n"
        "Rules (apply in order):\n"
        "1. User has replied -> useful\n"
        "2. Job alert email -> job_notifications\n"
        "3. Any possible future need -> useful\n"
        "4. Pure bulk marketing -> promotional\n"
        "5. Else -> need_further_review\n\n"
        "Input format (JSON array):\n"
        '[{"message_id": "id1", "from": "", "subject": "", "body": "", "user_replied": false}]\n\n'
        "Output only valid JSON with confidence and reason:\n"
        '{"results": [{"message_id": "id1", "category": "...", "confidence": "low|medium|high", '
        '"reason": "short reason"}]}\n\n'
        "Example:\n"
        'Input: [{"message_id": "123", "from": "jobs@linkedin.com", "subject": "New jobs for you", "user_replied": false}]\n'
        'Output: {"results": [{"message_id": "123", "category": "job_notifications", "confidence": "high", '
        '"reason": "LinkedIn job alert."}]}\n\n'
        f"Input: {json.dumps(emails, ensure_ascii=True)}"
    )


def _extract_json_object(text: str) -> dict[str, Any]:
    """Extract and parse the first JSON object contained in model text.
    Args:
        text: Raw model response text.
    Returns:
        Parsed JSON object as a dictionary.
    Raises:
        ValueError: Raised when no JSON object boundary is found.
        json.JSONDecodeError: Raised when the extracted text is invalid JSON.
    Used by:
        _classify_emails_with_qwen and _plan_gmail_mutation_with_qwen.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Model did not return a JSON object.")
    return json.loads(text[start : end + 1])


def _fallback_classification_results(
    emails: list[dict[str, Any]],
    reason: str,
) -> list[dict[str, Any]]:
    """Create conservative review rows when classifier JSON is unavailable.
    Args:
        emails: Classifier-ready email preview dictionaries.
        reason: Reason to store on each fallback review row.
    Returns:
        Review-result dictionaries marked need_further_review.
    Used by:
        _classify_emails_with_qwen when classifier JSON repair fails.
    """
    return [
        {
            "message_id": str(email.get("message_id") or ""),
            "thread_id": email.get("thread_id", ""),
            "date": email.get("date", ""),
            "from": email.get("from", ""),
            "subject": email.get("subject", ""),
            "snippet": email.get("snippet", ""),
            "category": "need_further_review",
            "confidence": "low",
            "reason": reason,
            "review_source": "fresh",
        }
        for email in emails
        if email.get("message_id")
    ]


# -----------------------------------------------------------------------------
# Gmail MCP Client Helpers
# -----------------------------------------------------------------------------
# This section builds the Gmail MCP client, loads MCP tools, invokes them, and coerces results.

def _build_gmail_mcp_client() -> MultiServerMCPClient:
    """Create the MCP client configuration for the local Gmail MCP server.
    Args:
        None.
    Returns:
        MultiServerMCPClient configured to run server.py over stdio.
    Used by:
        _load_gmail_mcp_tools when the cached MCP client is missing.
    """
    return MultiServerMCPClient(
        {
            "gmail": {
                "command": "python",
                "args": [str(MCP_SERVER_SCRIPT)],
                "transport": "stdio",
            }
        }
    )


async def _load_gmail_mcp_tools() -> dict[str, Any]:
    """Load and cache Gmail MCP tools from the local MCP server.
    Args:
        None.
    Returns:
        Dictionary mapping Gmail MCP tool names to tool objects.
    Side effects:
        Starts the MCP client on first use and emits a loading progress event.
    Used by:
        _invoke_gmail_mcp_tool and load_gmail_mutation_tools.
    """
    global _GMAIL_MCP_CLIENT, _GMAIL_MCP_TOOLS

    if _GMAIL_MCP_TOOLS is None:
        _emit_progress("loading_gmail_mcp_tools")
        _GMAIL_MCP_CLIENT = _build_gmail_mcp_client()
        tools = await _GMAIL_MCP_CLIENT.get_tools(server_name="gmail")
        _GMAIL_MCP_TOOLS = {tool.name: tool for tool in tools}
    return _GMAIL_MCP_TOOLS


async def _invoke_gmail_mcp_tool(tool_name: str, payload: dict[str, Any]) -> Any:
    """Invoke a named Gmail MCP tool with a payload.
    Args:
        tool_name: Name of the Gmail MCP tool to call.
        payload: Tool input payload.
    Returns:
        Raw tool result from the MCP adapter.
    Raises:
        ValueError: Raised when the requested MCP tool is unavailable.
    Used by:
        Gmail read helpers and confirmed Gmail mutation execution.
    """
    tools = await _load_gmail_mcp_tools()
    if tool_name not in tools:
        available = ", ".join(sorted(tools))
        raise ValueError(f"Gmail MCP tool not found: {tool_name}. Available: {available}")
    return await tools[tool_name].ainvoke(payload)


def _coerce_mcp_dict_result(result: Any, tool_name: str) -> dict[str, Any]:
    """Convert varied MCP adapter result shapes into a dictionary.
    Args:
        result: Raw MCP tool result, content list, text, or dictionary.
        tool_name: Tool name used for error messages.
    Returns:
        Parsed dictionary result.
    Raises:
        ValueError: Raised when the result cannot be interpreted as a dictionary.
    Used by:
        Gmail MCP wrapper helpers and mutation execution.
    """
    if isinstance(result, dict):
        return result

    content = getattr(result, "content", None)
    if content is not None:
        if isinstance(content, dict):
            return content
        if isinstance(content, list):
            return _coerce_mcp_dict_result(content, tool_name)
        if isinstance(content, str):
            return _coerce_mcp_dict_result(content, tool_name)

    if isinstance(result, list):
        for item in result:
            text = getattr(item, "text", None)
            if text is None and isinstance(item, dict):
                text = item.get("text")
            if text:
                return _coerce_mcp_dict_result(text, tool_name)
            if isinstance(item, dict) and item.get("type") != "text":
                return item
        raise ValueError(
            f"Gmail MCP tool {tool_name} returned an empty or unsupported content list."
        )

    if isinstance(result, str):
        text = result.strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(text)
            except (SyntaxError, ValueError) as exc:
                raise ValueError(
                    f"Gmail MCP tool {tool_name} returned non-JSON text: {text[:300]}"
                ) from exc
        if isinstance(parsed, dict):
            return parsed
        raise ValueError(
            f"Gmail MCP tool {tool_name} returned {type(parsed).__name__}, expected dict."
        )

    raise ValueError(
        f"Gmail MCP tool {tool_name} returned {type(result).__name__}, expected dict."
    )


async def _search_gmail_via_mcp(query: str, max_results: int) -> dict[str, Any]:
    """Search Gmail through the MCP search_gmail tool.
    Args:
        query: Gmail query string.
        max_results: Maximum number of message references to request.
    Returns:
        Dictionary result from the search_gmail MCP tool.
    Side effects:
        Emits a searching progress event and invokes the local MCP server.
    Used by:
        review_gmail_search.
    """
    _emit_progress("searching_gmail")
    result = await _invoke_gmail_mcp_tool(
        "search_gmail",
        {"query": query, "max_results": max_results},
    )
    return _coerce_mcp_dict_result(result, "search_gmail")


async def _search_gmail_date_window_via_mcp(
    base_query: str,
    after: str | None,
    before: str | None,
    max_results: int,
) -> dict[str, Any]:
    """Search Gmail through the MCP date-window search tool.
    Args:
        base_query: Base Gmail query to combine with dates.
        after: Optional Gmail after date in YYYY/MM/DD format.
        before: Optional Gmail before date in YYYY/MM/DD format.
        max_results: Maximum number of message references to request.
    Returns:
        Dictionary result from the search_gmail_date_window MCP tool.
    Side effects:
        Emits a searching progress event and invokes the local MCP server.
    Used by:
        review_gmail_date_window.
    """
    _emit_progress("searching_gmail")
    result = await _invoke_gmail_mcp_tool(
        "search_gmail_date_window",
        {
            "base_query": base_query,
            "after": after,
            "before": before,
            "max_results": max_results,
        },
    )
    return _coerce_mcp_dict_result(result, "search_gmail_date_window")


async def _get_gmail_message_via_mcp(message_id: str) -> dict[str, Any]:
    """Fetch one full Gmail message through the MCP read tool.
    Args:
        message_id: Gmail message ID to read.
    Returns:
        Normalized Gmail message dictionary from the MCP tool.
    Side effects:
        Emits a reading progress event and invokes the local MCP server.
    Used by:
        _review_message_refs for messages missing from cache.
    """
    _emit_progress("reading_gmail")
    result = await _invoke_gmail_mcp_tool(
        "get_gmail_message",
        {"message_id": message_id},
    )
    return _coerce_mcp_dict_result(result, "get_gmail_message")


# -----------------------------------------------------------------------------
# Email Classification And Review Cache Helpers
# -----------------------------------------------------------------------------
# This section classifies fetched emails and persists or reads cached review records.

async def _classify_emails_with_qwen(
    emails: list[dict[str, Any]],
    model: str = DEFAULT_MODEL,
) -> list[dict[str, Any]]:
    """Classify email previews with the configured local chat model.
    Args:
        emails: Classifier-ready email preview dictionaries.
        model: Model name to use for classification.
    Returns:
        List of normalized review-result dictionaries.
    Side effects:
        Emits classification progress and calls the local OpenAI-compatible model.
    Used by:
        _review_message_refs for freshly fetched emails.
    """
    if not emails:
        return []

    _emit_progress(
        "classifying_emails",
        f"Classifying {len(emails)} email{'s' if len(emails) != 1 else ''}...",
    )
    llm = build_llm(model=model, temperature=0.0, max_tokens=2_048)
    by_id = {str(email.get("message_id")): email for email in emails}
    classified: list[dict[str, Any]] = []

    for start in range(0, len(emails), MAX_CATEGORIZATION_BATCH):
        batch = emails[start : start + MAX_CATEGORIZATION_BATCH]
        prompt = _classification_prompt(batch)
        response = await llm.ainvoke(prompt)
        response_text = _message_text(response)
        try:
            payload = _extract_json_object(response_text)
        except (ValueError, json.JSONDecodeError):
            repair_response = await llm.ainvoke(
                (
                    f"{prompt}\n\n"
                    "The previous classifier output was invalid or blank:\n"
                    f"{response_text or '(blank)'}\n\n"
                    "Return one valid visible JSON object only."
                )
            )
            try:
                payload = _extract_json_object(_message_text(repair_response))
            except (ValueError, json.JSONDecodeError):
                classified.extend(
                    _fallback_classification_results(
                        batch,
                        "Classifier did not return valid visible JSON; needs manual review.",
                    )
                )
                continue
        for item in payload.get("results", []):
            message_id = str(item.get("message_id") or "")
            source = by_id.get(message_id, {})
            category = _category_from_legacy(item.get("category"))
            if category not in REVIEW_CATEGORIES:
                category = "need_further_review"
            confidence = str(item.get("confidence") or "low").lower()
            if confidence not in {"low", "medium", "high"}:
                confidence = "low"
            classified.append(
                {
                    "message_id": message_id,
                    "thread_id": source.get("thread_id", ""),
                    "date": source.get("date", ""),
                    "from": source.get("from", ""),
                    "subject": source.get("subject", ""),
                    "snippet": source.get("snippet", ""),
                    "category": category,
                    "confidence": confidence,
                    "reason": str(item.get("reason") or "No reason provided.").strip(),
                    "review_source": "fresh",
                }
            )

    classified_ids = {item["message_id"] for item in classified}
    for email in emails:
        message_id = str(email.get("message_id") or "")
        if message_id and message_id not in classified_ids:
            classified.append(
                {
                    "message_id": message_id,
                    "thread_id": email.get("thread_id", ""),
                    "date": email.get("date", ""),
                    "from": email.get("from", ""),
                    "subject": email.get("subject", ""),
                    "snippet": email.get("snippet", ""),
                    "category": "need_further_review",
                    "confidence": "low",
                    "reason": "Model did not return a classification for this message.",
                    "review_source": "fresh",
                }
            )
    return classified


def _load_review_db() -> dict[str, Any]:
    """Load the local Gmail review cache JSON file.
    Args:
        None.
    Returns:
        Cache payload dictionary with a messages mapping.
    Used by:
        Cache tools, stats, and _review_message_refs.
    """
    if not REVIEW_DB_FILE.exists():
        return {"messages": {}}
    try:
        payload = json.loads(REVIEW_DB_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"messages": {}}
    if not isinstance(payload, dict):
        return {"messages": {}}
    messages = payload.setdefault("messages", {})
    if not isinstance(messages, dict):
        payload["messages"] = {}
    return payload


def _save_review_db(payload: dict[str, Any]) -> None:
    """Persist the Gmail review cache payload to disk.
    Args:
        payload: Cache payload dictionary to write.
    Returns:
        None.
    Side effects:
        Updates the updated_at field and writes gmail_review_db.json.
    Used by:
        save_cached_email_reviews.
    """
    payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
    REVIEW_DB_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@tool
def get_cached_email_reviews(message_ids: list[str]) -> dict[str, Any]:
    """Load cached Gmail review results for specific message IDs.
    Args:
        message_ids: Gmail message IDs to look up in the local review cache.
    Returns:
        A dictionary with cached results, missing IDs, and cached/missing counts.
    Used by:
        Gmail read agent tools and _review_message_refs cache logic.
    """

    db = _load_review_db()
    messages = db.get("messages", {})
    cached = []
    missing = []
    for message_id in message_ids:
        row = messages.get(message_id)
        if row:
            cached.append(_normalize_cached_review(row))
        else:
            missing.append(message_id)
    return {
        "cached_count": len(cached),
        "missing_count": len(missing),
        "cached": cached,
        "missing_message_ids": missing,
    }


@tool
def save_cached_email_reviews(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Save Gmail review results to the local JSON cache.
    Args:
        results: Review-result dictionaries keyed by message_id or id.
    Returns:
        A dictionary with cache path, saved count, and total cached records.
    Side effects:
        Writes gmail_review_db.json.
    Used by:
        Gmail read agent tools and _review_message_refs after fresh classification.
    """

    db = _load_review_db()
    messages = db.setdefault("messages", {})
    saved = 0
    now = datetime.now().isoformat(timespec="seconds")
    for result in results:
        message_id = result.get("message_id") or result.get("id")
        if not message_id:
            continue
        existing = messages.get(message_id, {})
        row = {
            **existing,
            **result,
            "message_id": message_id,
            "category": _category_from_legacy(result.get("category") or result.get("classification")),
            "cached_at": now,
            "last_seen_at": now,
        }
        messages[message_id] = row
        saved += 1
    _save_review_db(db)
    return {
        "cache_path": str(REVIEW_DB_FILE),
        "saved_count": saved,
        "total_cached": len(messages),
    }


@tool
def get_review_cache_stats() -> dict[str, Any]:
    """Return counts and update metadata for the local Gmail review cache.
    Args:
        None.
    Returns:
        A dictionary with cache path, total count, category counts, and update time.
    Used by:
        Gmail read agent tools and diagnostics.
    """

    db = _load_review_db()
    messages = db.get("messages", {})
    counts = {
        category: 0
        for category in (*REVIEW_CATEGORIES, "unknown")
    }
    for row in messages.values():
        category = _category_from_legacy(row.get("category") or row.get("classification"))
        if category not in counts:
            category = "unknown"
        counts[category] += 1
    return {
        "cache_path": str(REVIEW_DB_FILE),
        "total_cached": len(messages),
        "category_counts": counts,
        "updated_at": db.get("updated_at"),
    }


def _bounded_max_results(max_results: int) -> int:
    """Clamp a requested scan limit to the supported tool range.
    Args:
        max_results: Requested maximum result count.
    Returns:
        Integer between 1 and MAX_TOOL_SCAN_LIMIT.
    Used by:
        review_gmail_search and review_gmail_date_window.
    """
    try:
        value = int(max_results)
    except (TypeError, ValueError):
        value = DEFAULT_SCAN_LIMIT
    return max(1, min(value, MAX_TOOL_SCAN_LIMIT))


# -----------------------------------------------------------------------------
# High-Level Gmail Review Tools
# -----------------------------------------------------------------------------
# This section defines agent tools that search Gmail, apply cache logic, classify, and save sessions.

async def _review_message_refs(
    refs: list[dict[str, Any]],
    scan_request: str,
    title: str,
    use_cache: bool,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """Review Gmail message refs using cache, fresh reads, classification, and session save.
    Args:
        refs: Gmail message reference dictionaries from search tools.
        scan_request: Human-readable scan request summary.
        title: Title for the saved review session.
        use_cache: Whether cached reviews may be reused.
        model: Model name used for fresh classification.
    Returns:
        Review summary with counts, session path, category counts, and ordered results.
    Side effects:
        Reads Gmail for cache misses and writes cache/session files.
    Used by:
        review_gmail_search and review_gmail_date_window.
    """
    message_ids = [ref.get("id") for ref in refs if ref.get("id")]
    _emit_progress("checking_review_cache")
    db = _load_review_db()
    cache = db.get("messages", {})
    cached_results: list[dict[str, Any]] = []
    missing_ids: list[str] = []

    for message_id in message_ids:
        cached = cache.get(message_id) if use_cache else None
        if cached:
            cached_results.append(_normalize_cached_review(cached))
        else:
            missing_ids.append(message_id)

    fetched_previews = []
    for message_id in missing_ids:
        email = await _get_gmail_message_via_mcp(message_id)
        fetched_previews.append(_email_preview_for_classification(email))

    fresh_results = await _classify_emails_with_qwen(fetched_previews, model=model)
    if fresh_results:
        _emit_progress("saving_review_cache")
        save_cached_email_reviews.invoke({"results": fresh_results})

    by_id = {
        item.get("message_id"): item
        for item in [*cached_results, *fresh_results]
        if item.get("message_id")
    }
    ordered_results = [by_id[message_id] for message_id in message_ids if message_id in by_id]

    _emit_progress("saving_review_session")
    saved_session = save_review_session.invoke(
        {
            "title": title,
            "scan_request": scan_request,
            "summary": (
                f"Reviewed {len(ordered_results)} Gmail messages. "
                f"Used cache for {len(cached_results)} and freshly classified {len(fresh_results)}."
            ),
            "results": ordered_results,
        }
    )
    return {
        "scan_request": scan_request,
        "result_count": len(ordered_results),
        "used_cache": use_cache,
        "cached_count": len(cached_results),
        "fresh_count": len(fresh_results),
        "missing_count": len(missing_ids),
        "session_path": saved_session["session_path"],
        "category_counts": saved_session["category_counts"],
        "results": ordered_results,
    }


@tool
async def review_gmail_search(
    query: str,
    max_results: int = DEFAULT_SCAN_LIMIT,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Review Gmail messages matching a query with cache-aware classification.
    Args:
        query: Gmail search query to review.
        max_results: Maximum number of message references to review.
        use_cache: Whether cached classifications may be reused.
    Returns:
        Review summary with counts, saved session path, and categorized results.
    Side effects:
        Searches Gmail, may read messages, writes cache records, and saves a session.
    Used by:
        Gmail read specialist as a high-level review tool.
    """

    bounded_max = _bounded_max_results(max_results)
    search_result = await _search_gmail_via_mcp(query=query, max_results=bounded_max)
    return await _review_message_refs(
        refs=search_result.get("messages", []),
        scan_request=f"query={query}; max_results={bounded_max}; use_cache={use_cache}",
        title=f"Gmail search {query}",
        use_cache=use_cache,
    )


@tool
async def review_gmail_date_window(
    base_query: str = "in:inbox category:primary",
    after: str | None = None,
    before: str | None = None,
    max_results: int = DEFAULT_SCAN_LIMIT,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Review Gmail messages in a date window with cache-aware classification.
    Args:
        base_query: Gmail query to combine with date filters.
        after: Optional Gmail after date in YYYY/MM/DD format.
        before: Optional Gmail before date in YYYY/MM/DD format.
        max_results: Maximum number of message references to review.
        use_cache: Whether cached classifications may be reused.
    Returns:
        Review summary with counts, saved session path, and categorized results.
    Side effects:
        Searches Gmail, may read messages, writes cache records, and saves a session.
    Used by:
        Gmail read specialist as a high-level date-window review tool.
    """

    bounded_max = _bounded_max_results(max_results)
    search_result = await _search_gmail_date_window_via_mcp(
        base_query=base_query,
        after=after,
        before=before,
        max_results=bounded_max,
    )
    query = search_result["query"]
    return await _review_message_refs(
        refs=search_result["messages"],
        scan_request=(
            f"query={query}; after={after}; before={before}; "
            f"max_results={bounded_max}; use_cache={use_cache}"
        ),
        title=f"Gmail date window {query}",
        use_cache=use_cache,
    )


# -----------------------------------------------------------------------------
# Review Session Tools
# -----------------------------------------------------------------------------
# This section defines tools for saving, listing, loading, and updating review-session files.

@tool
def save_review_session(
    title: str,
    scan_request: str,
    results: list[dict[str, Any]],
    summary: str = "",
) -> dict[str, Any]:
    """Save Gmail review results to a timestamped local session JSON file.
    Args:
        title: Human-readable session title.
        scan_request: Summary of the scan request that produced the results.
        results: Categorized Gmail review results to save.
        summary: Optional human-readable session summary.
    Returns:
        A dictionary with session path, result count, and category counts.
    Side effects:
        Creates review_sessions when needed and writes a JSON session file.
    Used by:
        _review_message_refs and Gmail read specialist tools.
    """

    REVIEW_SESSIONS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = "".join(
        char if char.isalnum() or char in ("-", "_") else "_"
        for char in title.strip().lower()
    ).strip("_")[:50]
    file_name = f"{timestamp}_{safe_title or 'gmail_review'}.json"
    path = REVIEW_SESSIONS_DIR / file_name
    payload = {
        "title": title,
        "scan_request": scan_request,
        "summary": summary,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "category_counts": {
            category: sum(1 for item in results if item.get("category") == category)
            for category in REVIEW_CATEGORIES
        },
        "results": results,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {
        "session_path": str(path),
        "result_count": len(results),
        "category_counts": payload["category_counts"],
    }


@tool
def list_review_sessions(max_results: int = 10) -> list[dict[str, Any]]:
    """List recent local Gmail review session files.
    Args:
        max_results: Maximum number of recent sessions to return.
    Returns:
        A list of session summaries with path, title, timestamp, and counts.
    Side effects:
        Ensures the review_sessions directory exists.
    Used by:
        Gmail read specialist tools and diagnostics.
    """

    REVIEW_SESSIONS_DIR.mkdir(exist_ok=True)
    files = sorted(
        REVIEW_SESSIONS_DIR.glob("*.json"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )[:max(1, max_results)]
    sessions = []
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        sessions.append(
            {
                "session_path": str(path),
                "title": payload.get("title", path.stem),
                "created_at": payload.get("created_at"),
                "category_counts": payload.get("category_counts", {}),
            }
        )
    return sessions


@tool
def load_review_session(session_path: str) -> dict[str, Any]:
    """Load a local Gmail review session JSON file.
    Args:
        session_path: Session file name or path inside review_sessions.
    Returns:
        Parsed review-session payload.
    Raises:
        ValueError: Raised when the path escapes review_sessions.
    Used by:
        Gmail read specialist tools when inspecting saved sessions.
    """

    path = _safe_session_path(session_path)
    return json.loads(path.read_text(encoding="utf-8"))


@tool
def update_review_category(
    session_path: str,
    message_id: str,
    category: Category,
    reason: str,
) -> dict[str, Any]:
    """Update one email category in a saved local review session.
    Args:
        session_path: Session file name or path inside review_sessions.
        message_id: Gmail message ID to update in the session.
        category: New review category to assign.
        reason: User-facing reason for the override.
    Returns:
        A dictionary with session path, updated message ID, category, and counts.
    Raises:
        ValueError: Raised when the path is unsafe or the message is not found.
    Side effects:
        Writes the updated review-session JSON file.
    Used by:
        Local review tools when users correct categorization results.
    """

    path = _safe_session_path(session_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    updated = None
    for item in payload.get("results", []):
        if item.get("message_id") == message_id or item.get("id") == message_id:
            item["category"] = category
            item["user_override"] = True
            item["override_reason"] = reason
            updated = item
            break

    if updated is None:
        raise ValueError(f"Message not found in session: {message_id}")

    payload["category_counts"] = {
        cat: sum(1 for item in payload.get("results", []) if item.get("category") == cat)
        for cat in REVIEW_CATEGORIES
    }
    payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {
        "session_path": str(path),
        "updated_message_id": message_id,
        "category": category,
        "category_counts": payload["category_counts"],
    }


# -----------------------------------------------------------------------------
# Tool Registration Lists
# -----------------------------------------------------------------------------
# This section groups local and Gmail mutation tools for the read and mutation agents.

GMAIL_READ_TOOLS = [
    review_gmail_search,
    review_gmail_date_window,
    get_cached_email_reviews,
    save_cached_email_reviews,
    get_review_cache_stats,
    save_review_session,
    list_review_sessions,
    load_review_session,
]

GMAIL_MUTATION_TOOL_NAMES = {
    "apply_gmail_label",
    "move_gmail_message_to_trash",
    "trash_gmail_messages_by_label",
}

LOCAL_REVIEW_TOOLS = [*GMAIL_READ_TOOLS, update_review_category]


# -----------------------------------------------------------------------------
# LLM And Agent Builders
# -----------------------------------------------------------------------------
# This section creates the chat model, loads tool sets, and builds specialist agent graphs.

def build_llm(
    model: str,
    temperature: float,
    max_tokens: int = 2_048,
) -> ChatOpenAI:
    """Build an OpenAI-compatible chat model client for LM Studio.
    Args:
        model: Model name served by the local OpenAI-compatible endpoint.
        temperature: Sampling temperature for responses.
        max_tokens: Maximum response tokens to request.
    Returns:
        Configured ChatOpenAI client.
    Used by:
        Classifier, router, specialist builders, final response node, and diagnostics.
    """
    return ChatOpenAI(
        model=model,
        base_url=LM_STUDIO_BASE_URL,
        api_key=LM_STUDIO_API_KEY,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=120,
    )


async def load_gmail_read_tools() -> list[Any]:
    """Return the local tools available to the Gmail read specialist.
    Args:
        None.
    Returns:
        List of Gmail read and review-session tools.
    Used by:
        load_tools and build_gmail_specialist_graph.
    """
    return GMAIL_READ_TOOLS


async def load_gmail_mutation_tools() -> list[Any]:
    """Load Gmail MCP mutation tools allowed for confirmed changes.
    Args:
        None.
    Returns:
        List of MCP tools whose names are in GMAIL_MUTATION_TOOL_NAMES.
    Side effects:
        Loads Gmail MCP tools on first use.
    Used by:
        Future mutation-agent tool-loading flows.
    """
    gmail_tools = await _load_gmail_mcp_tools()
    return [
        tool
        for name, tool in gmail_tools.items()
        if name in GMAIL_MUTATION_TOOL_NAMES
    ]


async def load_tools() -> list[Any]:
    """Return the default tool list for backward-compatible agent setup.
    Args:
        None.
    Returns:
        List of Gmail read tools.
    Used by:
        diagnostics and compatibility callers.
    """
    return await load_gmail_read_tools()


async def build_gmail_specialist_graph(model: str, temperature: float) -> Any:
    """Build the Gmail read specialist ReAct graph.
    Args:
        model: Model name served by LM Studio.
        temperature: Sampling temperature for the specialist LLM.
    Returns:
        Compiled ReAct agent runnable for Gmail read workflows.
    Used by:
        ask_gmail_read_agent.
    """
    tools = await load_gmail_read_tools()
    llm = build_llm(model=model, temperature=temperature, max_tokens=2_048)
    return create_react_agent(llm, tools, prompt=GMAIL_READ_SYSTEM_PROMPT)


async def build_agent_graph(model: str, temperature: float) -> Any:
    """Build the backward-compatible Gmail specialist subgraph.
    Args:
        model: Model name served by LM Studio.
        temperature: Sampling temperature for the specialist LLM.
    Returns:
        Compiled Gmail read specialist graph.
    Used by:
        Compatibility callers that still import build_agent_graph.
    """

    return await build_gmail_specialist_graph(model=model, temperature=temperature)


# -----------------------------------------------------------------------------
# Routing Helpers
# -----------------------------------------------------------------------------
# This section formats messages, compacts UI context, and routes each prompt to an agent.

def _message_text(message: Any) -> str:
    """Convert a LangChain message or raw content value to display text.
    Args:
        message: Message-like object or raw content.
    Returns:
        Stripped string content or JSON-formatted non-string content.
    Used by:
        Diagnostics, routing, classification, specialist calls, and final responses.
    """
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content.strip()
    return json.dumps(content, indent=2)


def compact_recent_ui_context(ui_messages: list[dict[str, str]], limit: int = 6) -> str:
    """Compress recent UI messages into a short router context string.
    Args:
        ui_messages: Streamlit-style chat messages with role and content.
        limit: Maximum number of recent messages to include.
    Returns:
        Newline-delimited compact context string.
    Used by:
        route_prompt.
    """
    recent = ui_messages[-limit:]
    return "\n".join(
        f"{message.get('role', 'unknown')}: {str(message.get('content', ''))[:500]}"
        for message in recent
    )


def _router_state_context(state: AgentState) -> str:
    """Serialize key agent state fields for the router prompt.
    Args:
        state: Current agent state.
    Returns:
        JSON string summarizing route, review, and pending-action state.
    Used by:
        route_prompt.
    """
    return json.dumps(
        {
            "last_route": state.last_route,
            "has_review_context": _has_review_context(state),
            "latest_review_count": len(state.latest_review_results),
            "has_pending_action": bool(state.pending_action),
            "pending_action_type": (state.pending_action or {}).get("action_type"),
        },
        ensure_ascii=True,
    )


def _router_user_message(
    prompt: str,
    state: AgentState,
    ui_messages: list[dict[str, str]] | None,
) -> str:
    """Build the user message sent to the router model.
    Args:
        prompt: Current user message.
        state: Current agent state.
        ui_messages: Optional recent UI messages for routing context.
    Returns:
        Router prompt content with recent UI context, state, and user message.
    Used by:
        route_prompt initial and repair routing calls.
    """
    return (
        "Recent UI context:\n"
        f"{compact_recent_ui_context(ui_messages or []) or '(none)'}\n\n"
        "Structured app state:\n"
        f"{_router_state_context(state)}\n\n"
        f"Current user message:\n/no_think\n{prompt}\n\n"
        "Return the route as visible JSON only."
    )


def _parse_router_payload(text: str) -> tuple[Route, str] | None:
    """Parse a router JSON response into a route and reason.
    Args:
        text: Raw visible model response text.
    Returns:
        Route and reason tuple, or None when parsing or validation fails.
    Used by:
        route_prompt after initial and repair routing calls.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None

    route = payload.get("route")
    if route not in {"chat_agent", "gmail_read_agent", "gmail_mutation_agent"}:
        return None
    return route, str(payload.get("reason", "router decision"))


async def route_prompt(
    prompt: str,
    state: AgentState,
    model: str,
    ui_messages: list[dict[str, str]] | None = None,
) -> tuple[Route, str]:
    """Route a user prompt to chat, Gmail read, or Gmail mutation handling.
    Args:
        prompt: Current user message.
        state: Current agent state.
        model: Model name used by the router LLM.
        ui_messages: Optional recent UI messages for routing context.
    Returns:
        Tuple of selected route and short route reason.
    Side effects:
        Calls the local OpenAI-compatible model and may retry invalid router JSON.
    Used by:
        router_node.
    """
    llm = build_llm(model=model, temperature=0.0, max_tokens=256)
    router_message = _router_user_message(prompt, state, ui_messages)
    response = await llm.ainvoke(
        [
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=router_message),
        ]
    )
    text = _message_text(response)
    parsed = _parse_router_payload(text)
    if parsed:
        return parsed

    repair_response = await llm.ainvoke(
        [
            SystemMessage(content=ROUTER_REPAIR_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"{router_message}\n\n"
                    "The previous router output was invalid or blank:\n"
                    f"{text or '(blank)'}\n\n"
                    "Return valid visible JSON only."
                )
            ),
        ]
    )
    repaired = _parse_router_payload(_message_text(repair_response))
    if repaired:
        route, reason = repaired
        return route, f"{reason} (router JSON repaired)"
    return "chat_agent", "router returned invalid JSON after repair"


# -----------------------------------------------------------------------------
# Gmail Read Agent Runtime
# -----------------------------------------------------------------------------
# This section runs the Gmail read specialist and refreshes latest review-session context.

def _latest_review_session_payload() -> tuple[dict[str, Any] | None, str | None]:
    """Load the newest review-session payload and path, if one exists.
    Args:
        None.
    Returns:
        Tuple of session payload and path string, or None values when unavailable.
    Side effects:
        Ensures the review_sessions directory exists.
    Used by:
        gmail_read_specialist_node after read-agent work completes.
    """
    REVIEW_SESSIONS_DIR.mkdir(exist_ok=True)
    files = sorted(
        REVIEW_SESSIONS_DIR.glob("*.json"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not files:
        return None, None
    path = files[0]
    try:
        return json.loads(path.read_text(encoding="utf-8")), str(path)
    except json.JSONDecodeError:
        return None, str(path)


async def ask_gmail_read_agent(
    prompt: str,
    state: AgentState,
    model: str,
    temperature: float,
) -> tuple[str, list[BaseMessage]]:
    """Run the Gmail read specialist for a user prompt.
    Args:
        prompt: User request to pass to the read specialist.
        state: Current agent state containing prior Gmail read messages.
        model: Model name used by the specialist.
        temperature: Sampling temperature used by the specialist.
    Returns:
        Tuple of response text and updated Gmail specialist message history.
    Side effects:
        Calls the local LLM and may invoke Gmail read/review tools through the agent.
    Used by:
        gmail_read_specialist_node.
    """
    graph = await build_gmail_specialist_graph(model=model, temperature=temperature)
    messages = [*state.gmail_messages, HumanMessage(content=prompt)]
    result = await graph.ainvoke({"messages": messages})
    gmail_messages = result["messages"]
    last_ai = next(
        (
            message
            for message in reversed(gmail_messages)
            if isinstance(message, AIMessage)
        ),
        None,
    )
    return (
        _message_text(last_ai) if last_ai else "I did not receive a model response.",
        gmail_messages,
    )


# -----------------------------------------------------------------------------
# Gmail Mutation Agent Runtime
# -----------------------------------------------------------------------------
# This section plans, validates, confirms, and executes Gmail label or trash actions.

def _has_review_context(state: AgentState) -> bool:
    """Check whether mutation planning has reviewed Gmail context available.
    Args:
        state: Current agent state.
    Returns:
        True when latest review results or a session path are available.
    Used by:
        Router state serialization and ask_gmail_mutation_agent.
    """
    return bool(state.latest_review_results or state.latest_review_session_path)


def _mutation_blocked_output() -> str:
    """Build the JSON response used when Gmail mutation is blocked.
    Args:
        None.
    Returns:
        JSON string explaining that review context is required first.
    Used by:
        ask_gmail_mutation_agent when no review context exists.
    """
    return json.dumps(
        {
            "status": "blocked",
            "reason": "mutation_requires_review_context",
            "message": "Gmail changes cannot be made until emails have been scanned and reviewed.",
            "suggested_next_steps": [
                "Ask me to scan Primary inbox first.",
                "Include a date range and maximum number of emails.",
                "Example: Scan Primary inbox from 2026/05/02 to 2026/05/03, max 10.",
            ],
        },
        ensure_ascii=True,
    )


def _review_context_for_planner(results: list[dict[str, Any]], limit: int = 60) -> list[dict[str, Any]]:
    """Create compact reviewed-message context for the mutation planner.
    Args:
        results: Latest Gmail review result rows.
        limit: Maximum number of rows to include.
    Returns:
        List of compact message dictionaries with IDs and classification context.
    Used by:
        _mutation_planner_prompt.
    """
    return [
        {
            "message_id": row.get("message_id") or row.get("id"),
            "category": row.get("category"),
            "from": row.get("from", ""),
            "subject": row.get("subject", ""),
            "reason": row.get("reason", ""),
        }
        for row in results[:limit]
        if row.get("message_id") or row.get("id")
    ]


def _mutation_planner_prompt(
    prompt: str,
    results: list[dict[str, Any]],
    pending_action: dict[str, Any] | None,
) -> str:
    """Build the JSON-only prompt for planning a Gmail mutation.
    Args:
        prompt: User mutation request.
        results: Latest reviewed Gmail results.
        pending_action: Existing pending action, if any.
    Returns:
        Prompt string instructing the model to produce a mutation plan JSON object.
    Used by:
        _plan_gmail_mutation_with_qwen.
    """
    return (
        "/no_think\n"
        "Interpret the user's Gmail mutation request and return only valid JSON.\n\n"
        "You may plan one of these action_type values:\n"
        "- apply_labels: apply a Gmail label to reviewed messages\n"
        "- trash_messages: move reviewed messages to Trash\n"
        "- cancel: cancel the pending action\n"
        "- needs_clarification: request clarification\n\n"
        "Important rules:\n"
        "- If the user asks to label/apply a label/tag messages, use apply_labels even if the label text contains words like delete or trash.\n"
        "- Preserve custom label names exactly, especially quoted text like \"To_delete\" or \"To be deleted\".\n"
        "- Only target message_id values present in review_results.\n"
        "- If the user confirms a pending action, set status to confirmed_existing_action.\n"
        "- If the user asks for a new action, set status to requires_confirmation.\n"
        "- Never execute anything. Only plan.\n\n"
        "Return this JSON shape:\n"
        "{"
        '"status":"requires_confirmation|confirmed_existing_action|cancelled|needs_clarification",'
        '"action_type":"apply_labels|trash_messages|cancel|needs_clarification",'
        '"label_name":"label to apply or null",'
        '"target_message_ids":["..."],'
        '"user_facing_summary":"short summary",'
        '"question":"clarifying question or null"'
        "}\n\n"
        f"pending_action:\n{json.dumps(pending_action, ensure_ascii=True)}\n\n"
        f"review_results:\n{json.dumps(_review_context_for_planner(results), ensure_ascii=True)}\n\n"
        f"user_request:\n{prompt}\n"
        "/no_think"
    )


def _target_rows_by_id(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index reviewed result rows by Gmail message ID.
    Args:
        results: Review result dictionaries.
    Returns:
        Dictionary mapping message IDs to their source result rows.
    Used by:
        _validated_pending_action_from_plan.
    """
    return {
        str(row.get("message_id") or row.get("id")): row
        for row in results
        if row.get("message_id") or row.get("id")
    }


def _validated_pending_action_from_plan(
    plan: dict[str, Any],
    state: AgentState,
) -> dict[str, Any]:
    """Validate a model-planned Gmail mutation against latest review results.
    Args:
        plan: Parsed mutation plan from the local model.
        state: Current agent state containing latest review results.
    Returns:
        Pending action dictionary safe to preview and later execute.
    Raises:
        ValueError: Raised when action type, label, targets, or IDs are invalid.
    Used by:
        ask_gmail_mutation_agent when a new action needs confirmation.
    """
    action_type = plan.get("action_type")
    if action_type not in {"apply_labels", "trash_messages"}:
        raise ValueError(f"Unsupported Gmail mutation action: {action_type}")
    label_name = plan.get("label_name")
    if action_type == "apply_labels" and not str(label_name or "").strip():
        raise ValueError("A Gmail label action requires a label_name.")

    by_id = _target_rows_by_id(state.latest_review_results)
    target_ids = [str(message_id) for message_id in plan.get("target_message_ids", [])]
    targets = []
    rejected_ids = []
    for message_id in target_ids:
        row = by_id.get(message_id)
        if not row:
            rejected_ids.append(message_id)
            continue
        targets.append(
            {
                "message_id": message_id,
                "subject": row.get("subject", ""),
                "from": row.get("from", ""),
                "category": row.get("category"),
                "label_name": str(label_name).strip() if action_type == "apply_labels" else None,
            }
        )
    if rejected_ids:
        raise ValueError(
            "Mutation plan referenced messages outside the latest review results: "
            + ", ".join(rejected_ids)
        )
    if not targets:
        raise ValueError("Mutation plan did not select any reviewed messages.")

    return {
        "action_type": action_type,
        "source_session_path": state.latest_review_session_path,
        "targets": targets,
        "user_facing_summary": plan.get("user_facing_summary", ""),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }


async def _plan_gmail_mutation_with_qwen(
    prompt: str,
    state: AgentState,
    model: str,
    temperature: float,
) -> dict[str, Any]:
    """Ask the local model to plan a Gmail mutation from reviewed results.
    Args:
        prompt: User mutation request.
        state: Current agent state with review context and pending action.
        model: Model name used by the planner.
        temperature: Temperature parameter kept for caller interface consistency.
    Returns:
        Parsed mutation plan dictionary.
    Side effects:
        Calls the local OpenAI-compatible model.
    Used by:
        ask_gmail_mutation_agent.
    """
    llm = build_llm(model=model, temperature=0.0, max_tokens=512)
    response = await llm.ainvoke(
        _mutation_planner_prompt(
            prompt=prompt,
            results=state.latest_review_results,
            pending_action=state.pending_action,
        )
    )
    return _extract_json_object(_message_text(response))


async def _execute_pending_gmail_action(pending_action: dict[str, Any]) -> dict[str, Any]:
    """Execute a confirmed Gmail label or trash pending action.
    Args:
        pending_action: Validated pending action with action type and targets.
    Returns:
        Execution summary with completed and failed message results.
    Raises:
        ValueError: Raised when the pending action type is unsupported.
    Side effects:
        Applies Gmail labels or moves Gmail messages to Trash via MCP tools.
    Used by:
        ask_gmail_mutation_agent after user confirmation.
    """
    _emit_progress("executing_gmail_mutation")
    action_type = pending_action.get("action_type")
    targets = [
        target
        for target in pending_action.get("targets", [])
        if target.get("message_id")
    ]

    completed = []
    failed = []
    if action_type == "apply_labels":
        for target in targets:
            try:
                result = await _invoke_gmail_mcp_tool(
                    "apply_gmail_label",
                    {
                        "message_id": target["message_id"],
                        "label_name": target.get("label_name") or target.get("category"),
                        "create_if_missing": True,
                    },
                )
                completed.append(
                    _coerce_mcp_dict_result(result, "apply_gmail_label")
                )
            except Exception as exc:
                failed.append(
                    {
                        "message_id": target["message_id"],
                        "error": str(exc),
                    }
                )
    elif action_type == "trash_messages":
        for target in targets:
            try:
                result = await _invoke_gmail_mcp_tool(
                    "move_gmail_message_to_trash",
                    {"message_id": target["message_id"]},
                )
                completed.append(
                    _coerce_mcp_dict_result(result, "move_gmail_message_to_trash")
                )
            except Exception as exc:
                failed.append(
                    {
                        "message_id": target["message_id"],
                        "error": str(exc),
                    }
                )
    else:
        raise ValueError(f"Unsupported pending Gmail action: {action_type}")

    return {
        "status": "executed",
        "action_type": action_type,
        "requested_count": len(targets),
        "completed_count": len(completed),
        "failed_count": len(failed),
        "completed": completed,
        "failed": failed,
    }


async def ask_gmail_mutation_agent(
    prompt: str,
    state: AgentState,
    model: str,
    temperature: float,
) -> tuple[str, list[BaseMessage], dict[str, Any] | None]:
    """Plan, confirm, cancel, or execute Gmail mutations using reviewed context.
    Args:
        prompt: User mutation request or confirmation reply.
        state: Current agent state with review context and pending action.
        model: Model name used by the mutation planner.
        temperature: Sampling temperature passed through the caller interface.
    Returns:
        Tuple of JSON response text, updated mutation messages, and pending action.
    Side effects:
        May execute confirmed Gmail label or Trash changes through MCP tools.
    Used by:
        gmail_mutation_specialist_node.
    """
    messages = [*state.gmail_mutation_messages, HumanMessage(content=prompt)]

    if not _has_review_context(state):
        output = _mutation_blocked_output()
        return output, [*messages, AIMessage(content=output)], state.pending_action

    plan = await _plan_gmail_mutation_with_qwen(
        prompt=prompt,
        state=state,
        model=model,
        temperature=temperature,
    )
    status = plan.get("status")
    if status == "cancelled" or plan.get("action_type") == "cancel":
        output = json.dumps(
            {
                "status": "cancelled",
                "message": "The pending Gmail mutation was cancelled.",
            },
            ensure_ascii=True,
        )
        return output, [*messages, AIMessage(content=output)], None

    if status == "confirmed_existing_action":
        if not state.pending_action:
            output = json.dumps(
                {
                    "status": "blocked",
                    "reason": "no_pending_action",
                    "message": "There is no pending Gmail action to confirm.",
                },
                ensure_ascii=True,
            )
            return output, [*messages, AIMessage(content=output)], None
        execution_result = await _execute_pending_gmail_action(state.pending_action)
        output = json.dumps(execution_result, ensure_ascii=True)
        return output, [*messages, AIMessage(content=output)], None

    if status == "needs_clarification":
        output = json.dumps(
            {
                "status": "needs_clarification",
                "question": plan.get("question") or "Can you clarify which reviewed emails to change?",
            },
            ensure_ascii=True,
        )
        return output, [*messages, AIMessage(content=output)], state.pending_action

    pending_action = _validated_pending_action_from_plan(plan, state)
    output = json.dumps(
        {
            "status": "requires_confirmation",
            "message": plan.get("user_facing_summary") or "Prepared a Gmail action from the latest review results.",
            "action_type": pending_action["action_type"],
            "label_name": pending_action["targets"][0].get("label_name"),
            "target_count": len(pending_action["targets"]),
            "preview": pending_action["targets"][:10],
            "confirmation_prompt": (
                "Reply confirm to move these messages to Trash, or cancel to stop."
                if pending_action["action_type"] == "trash_messages"
                else "Reply confirm to apply these labels, or cancel to stop."
            ),
        },
        ensure_ascii=True,
    )
    return output, [*messages, AIMessage(content=output)], pending_action


# -----------------------------------------------------------------------------
# Workflow Graph Nodes
# -----------------------------------------------------------------------------
# This section implements LangGraph nodes for routing, specialists, and final responses.

def _specialist_context(state: WorkflowState) -> str:
    """Format specialist output for the final response model.
    Args:
        state: Current workflow state.
    Returns:
        Formatted specialist context string, or an empty string when absent.
    Used by:
        final_response_node.
    """
    specialist_name = state.get("specialist_name")
    specialist_output = state.get("specialist_output")
    if not specialist_name or not specialist_output:
        return ""
    return (
        f"Specialist used: {specialist_name}\n"
        f"Specialist result:\n{specialist_output}"
    )


async def router_node(state: WorkflowState) -> dict[str, Any]:
    """LangGraph node that routes the current prompt to the next agent path.
    Args:
        state: Current workflow state.
    Returns:
        Partial workflow state containing route, reason, and specialist name.
    Side effects:
        Emits routing progress and calls route_prompt.
    Used by:
        build_workflow_graph.
    """
    _emit_progress("routing")
    route, route_reason = await route_prompt(
        prompt=state["prompt"],
        state=state["agent_state"],
        model=state["model"],
        ui_messages=state.get("ui_messages"),
    )
    return {
        "route": route,
        "route_reason": route_reason,
        "specialist_name": route.replace("_agent", "") if route != "chat_agent" else None,
    }


def route_after_router(state: WorkflowState) -> str:
    """Choose the next LangGraph node after routing.
    Args:
        state: Workflow state containing the selected route.
    Returns:
        Node name for the next graph step.
    Used by:
        build_workflow_graph conditional routing.
    """
    if state.get("route") == "gmail_read_agent":
        return "gmail_read_specialist"
    if state.get("route") == "gmail_mutation_agent":
        return "gmail_mutation_specialist"
    return "final_response"


async def gmail_read_specialist_node(state: WorkflowState) -> dict[str, Any]:
    """LangGraph node that runs the Gmail read specialist.
    Args:
        state: Current workflow state.
    Returns:
        Partial workflow state with specialist output, messages, and review context.
    Side effects:
        Emits progress and may trigger Gmail read/review tools through the specialist.
    Used by:
        build_workflow_graph.
    """
    _emit_progress("gmail_read_agent_at_work")
    response, gmail_messages = await ask_gmail_read_agent(
        prompt=state["prompt"],
        state=state["agent_state"],
        model=state["model"],
        temperature=state["temperature"],
    )
    session, session_path = _latest_review_session_payload()
    latest_results = (
        session.get("results", [])
        if session
        else state["agent_state"].latest_review_results
    )
    return {
        "specialist_name": "gmail_read",
        "specialist_output": response,
        "gmail_messages": gmail_messages,
        "latest_review_session_path": session_path or state["agent_state"].latest_review_session_path,
        "latest_review_results": latest_results,
    }


async def gmail_mutation_specialist_node(state: WorkflowState) -> dict[str, Any]:
    """LangGraph node that runs the Gmail mutation specialist.
    Args:
        state: Current workflow state.
    Returns:
        Partial workflow state with mutation output, messages, and pending action.
    Side effects:
        Emits progress and may execute confirmed Gmail mutations.
    Used by:
        build_workflow_graph.
    """
    _emit_progress("gmail_mutation_agent_at_work")
    response, mutation_messages, pending_action = await ask_gmail_mutation_agent(
        prompt=state["prompt"],
        state=state["agent_state"],
        model=state["model"],
        temperature=state["temperature"],
    )
    return {
        "specialist_name": "gmail_mutation",
        "specialist_output": response,
        "gmail_mutation_messages": mutation_messages,
        "pending_action": pending_action,
    }


async def final_response_node(state: WorkflowState) -> dict[str, Any]:
    """LangGraph node that generates the final user-facing response.
    Args:
        state: Current workflow state with optional specialist output.
    Returns:
        Partial workflow state containing response text and next AgentState.
    Side effects:
        Emits progress and calls the local OpenAI-compatible model.
    Used by:
        build_workflow_graph.
    """
    _emit_progress("preparing_response")
    current_prompt = state["prompt"]
    prior_messages = state["agent_state"].chat_messages
    specialist_context = _specialist_context(state)

    if specialist_context:
        user_content = (
            f"User request:\n{current_prompt}\n\n"
            f"{specialist_context}\n\n"
            "Answer the user using the specialist result as context."
        )
        max_tokens = 512
    else:
        user_content = f"/no_think\n{current_prompt}"
        max_tokens = 256

    llm = build_llm(
        model=state["model"],
        temperature=state["temperature"],
        max_tokens=max_tokens,
    )
    response_message = await llm.ainvoke(
        [
            SystemMessage(content=FINAL_RESPONSE_SYSTEM_PROMPT),
            *prior_messages,
            HumanMessage(content=user_content),
        ]
    )
    response_text = _message_text(response_message)

    specialist_outputs = list(state["agent_state"].last_specialist_outputs)
    if specialist_context:
        specialist_outputs.append(
            {
                "specialist": state.get("specialist_name"),
                "route": state.get("route"),
                "output": state.get("specialist_output"),
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
        )

    next_agent_state = AgentState(
        chat_messages=[
            *prior_messages,
            HumanMessage(content=current_prompt),
            AIMessage(content=response_text),
        ],
        gmail_messages=state.get("gmail_messages", state["agent_state"].gmail_messages),
        gmail_mutation_messages=state.get(
            "gmail_mutation_messages",
            state["agent_state"].gmail_mutation_messages,
        ),
        latest_review_session_path=state.get(
            "latest_review_session_path",
            state["agent_state"].latest_review_session_path,
        ),
        latest_review_results=state.get(
            "latest_review_results",
            state["agent_state"].latest_review_results,
        ),
        pending_action=state.get("pending_action", state["agent_state"].pending_action),
        last_specialist_outputs=specialist_outputs[-20:],
        last_route=state.get("route"),
    )
    return {
        "response": response_text,
        "next_agent_state": next_agent_state,
    }


# -----------------------------------------------------------------------------
# Workflow Graph Construction And Public Entrypoints
# -----------------------------------------------------------------------------
# This section compiles the LangGraph workflow and exposes the main async message handler.

async def build_workflow_graph() -> Any:
    """Build and compile the LangGraph workflow for one user turn.
    Args:
        None.
    Returns:
        Compiled workflow graph runnable.
    Used by:
        handle_user_message.
    """
    graph = StateGraph(WorkflowState)
    graph.add_node("router", router_node)
    graph.add_node("gmail_read_specialist", gmail_read_specialist_node)
    graph.add_node("gmail_mutation_specialist", gmail_mutation_specialist_node)
    graph.add_node("final_response", final_response_node)

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "gmail_read_specialist": "gmail_read_specialist",
            "gmail_mutation_specialist": "gmail_mutation_specialist",
            "final_response": "final_response",
        },
    )
    graph.add_edge("gmail_read_specialist", "final_response")
    graph.add_edge("gmail_mutation_specialist", "final_response")
    graph.add_edge("final_response", END)
    return graph.compile()


async def handle_user_message(
    prompt: str,
    state: AgentState,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.1,
    ui_messages: list[dict[str, str]] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> AgentTurnResult:
    """Handle one user message through routing, specialists, and final response.
    Args:
        prompt: Current user message.
        state: Current persistent agent state.
        model: Model name used by router, specialists, and final response.
        temperature: Sampling temperature for non-deterministic response calls.
        ui_messages: Optional recent UI messages for routing context.
        progress_callback: Optional callback receiving progress stage and label.
    Returns:
        AgentTurnResult with response text, updated state, route, and route reason.
    Side effects:
        May read Gmail, write review files, or execute confirmed Gmail mutations.
    Used by:
        Streamlit UI and run_chat.
    """
    token = _PROGRESS_CALLBACK.set(progress_callback)
    try:
        graph = await build_workflow_graph()
        result = await graph.ainvoke(
            {
                "prompt": prompt,
                "agent_state": state,
                "model": model,
                "temperature": temperature,
                "ui_messages": ui_messages or [],
            }
        )
        _emit_progress("done")
    finally:
        _PROGRESS_CALLBACK.reset(token)

    return AgentTurnResult(
        response=result["response"],
        state=result["next_agent_state"],
        route=result["route"],
        route_reason=result["route_reason"],
    )


# -----------------------------------------------------------------------------
# CLI Entrypoint
# -----------------------------------------------------------------------------
# This section runs the interactive terminal chat loop when this module is executed.

async def run_chat(model: str, temperature: float) -> None:
    """Run the interactive terminal chat loop.
    Args:
        model: Model name used by the agent workflow.
        temperature: Sampling temperature used by the agent workflow.
    Returns:
        None.
    Side effects:
        Reads from stdin, prints responses, and may trigger workflow side effects.
    Used by:
        main when this module is executed as a script.
    """
    state = AgentState()

    print("Gmail agent ready. Type 'exit' or 'quit' to stop.")
    print(f"Model: {model} | context target: {CONTEXT_WINDOW_TOKENS} | batch target: {MAX_CATEGORIZATION_BATCH} emails")

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.casefold() in {"exit", "quit"}:
            print("Bye.")
            return

        result = await handle_user_message(
            prompt=user_input,
            state=state,
            model=model,
            temperature=temperature,
        )
        state = result.state
        print(f"\nAgent: {result.response}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the terminal chat agent.
    Args:
        None.
    Returns:
        Parsed argparse namespace with model and temperature.
    Used by:
        main in this module.
    """
    parser = argparse.ArgumentParser(description="Local LM Studio Gmail MCP agent.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    """Start the terminal chat agent from parsed command-line arguments.
    Args:
        None.
    Returns:
        None.
    Side effects:
        Runs the async terminal chat loop.
    Used by:
        command-line execution of this module.
    """
    args = parse_args()
    asyncio.run(run_chat(model=args.model, temperature=args.temperature))


if __name__ == "__main__":
    main()
