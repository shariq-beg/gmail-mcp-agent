# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
# This section imports Streamlit, data, path, async, and agent helpers used by the UI.

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from agent import (
    AgentState,
    CONTEXT_WINDOW_TOKENS,
    DEFAULT_MODEL,
    MAX_CATEGORIZATION_BATCH,
    PROGRESS_STAGE_LABELS,
    REVIEW_CATEGORIES,
    REVIEW_DB_FILE,
    REVIEW_SESSIONS_DIR,
    handle_user_message,
)


# -----------------------------------------------------------------------------
# UI Text And Progress Configuration
# -----------------------------------------------------------------------------
# This section defines user-facing progress messages and the initial assistant greeting.


CHAT_PROGRESS_MESSAGES = {
    "routing": "Okay, I am working out who should handle this.",
    "loading_gmail_mcp_tools": "I am connecting to the Gmail tools.",
    "gmail_read_agent_at_work": "I have activated the Gmail read agent.",
    "gmail_mutation_agent_at_work": "I have activated the Gmail mutation agent.",
    "searching_gmail": "I am searching Gmail.",
    "reading_gmail": "I am reading the matching emails.",
    "checking_review_cache": "I am checking what I have already reviewed.",
    "classifying_emails": "I am classifying the emails.",
    "saving_review_cache": "I am saving the review cache.",
    "saving_review_session": "I am saving the review results.",
    "executing_gmail_mutation": "I am applying the confirmed Gmail changes.",
    "preparing_response": "I am preparing the answer for you.",
    "done": "Done.",
}

WELCOME_MESSAGE = (
    "Hi I am Qwen, your personal assistant. How can I help you today?\n\n"
    "Here are my current capabilities:\n"
    "1. I can read your Gmails and help you clean it up or summarise it."
)


# -----------------------------------------------------------------------------
# Streamlit Page Setup
# -----------------------------------------------------------------------------
# This section configures the Streamlit page title, icon, and layout before rendering.

st.set_page_config(
    page_title="Gmail Review Agent",
    page_icon="",
    layout="wide",
)


# -----------------------------------------------------------------------------
# Session State Helpers
# -----------------------------------------------------------------------------
# This section initializes and resets chat, model, review, and agent state for the UI.

def run_async(value):
    """Run an awaitable to completion from Streamlit's synchronous script flow.
    Args:
        value: Awaitable object to execute.
    Returns:
        The awaitable's resolved result.
    Used by:
        App Layout when calling handle_user_message.
    """
    return asyncio.run(value)


def reset_chat_state() -> None:
    """Reset visible chat, agent state, latest review data, and notices.
    Args:
        None.
    Returns:
        None.
    Side effects:
        Mutates st.session_state values used by the chat and review UI.
    Used by:
        ensure_state during first load and the New Chat button.
    """
    st.session_state.ui_messages = [
        {
            "role": "assistant",
            "content": WELCOME_MESSAGE,
        }
    ]
    st.session_state.agent_state = AgentState()
    st.session_state.latest_session = None
    st.session_state.latest_session_path = None
    st.session_state.review_notice = None


def ensure_state() -> None:
    """Ensure all required Streamlit session-state keys exist.
    Args:
        None.
    Returns:
        None.
    Side effects:
        Initializes missing st.session_state keys with default values.
    Used by:
        App Layout before rendering UI controls.
    """
    if "model" not in st.session_state:
        st.session_state.model = DEFAULT_MODEL
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.1
    if "ui_messages" not in st.session_state:
        reset_chat_state()
    if "agent_state" not in st.session_state:
        st.session_state.agent_state = AgentState()
    if "latest_session" not in st.session_state:
        st.session_state.latest_session = None
    if "latest_session_path" not in st.session_state:
        st.session_state.latest_session_path = None
    if "review_notice" not in st.session_state:
        st.session_state.review_notice = None


def category_from_legacy(value: Any) -> str:
    """Normalize older category names into the current category set.
    Args:
        value: Category or classification value to normalize.
    Returns:
        Normalized category string, or unknown for blank values.
    Used by:
        load_cache_stats and session_results when reading persisted data.
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
    return aliases.get(normalized, normalized or "unknown")


# -----------------------------------------------------------------------------
# Cache And Review Session Helpers
# -----------------------------------------------------------------------------
# This section reads local cache and review-session files for sidebar and table display.

def load_cache_stats() -> dict[str, Any]:
    """Load summary counts from the local Gmail review cache file.
    Args:
        None.
    Returns:
        A dictionary with total cached messages, category counts, and update time.
    Used by:
        App Layout when rendering the Persistent Cache sidebar panel.
    """
    if not REVIEW_DB_FILE.exists():
        return {
            "total_cached": 0,
            "category_counts": {},
            "updated_at": None,
        }
    try:
        payload = json.loads(REVIEW_DB_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {
            "total_cached": 0,
            "category_counts": {},
            "updated_at": "Unreadable cache file",
        }

    messages = payload.get("messages", {})
    counts = {
        **{category: 0 for category in REVIEW_CATEGORIES},
        "unknown": 0,
    }
    for row in messages.values():
        category = category_from_legacy(row.get("category") or row.get("classification"))
        if category not in counts:
            category = "unknown"
        counts[category] += 1

    return {
        "total_cached": len(messages),
        "category_counts": counts,
        "updated_at": payload.get("updated_at"),
    }


def latest_review_session() -> tuple[dict[str, Any] | None, Path | None]:
    """Load the most recently modified review-session JSON file.
    Args:
        None.
    Returns:
        A tuple of session payload and path, or None values when unavailable.
    Side effects:
        Ensures the review sessions directory exists.
    Used by:
        latest_review_session_if_newer after Gmail read-agent turns.
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
        return json.loads(path.read_text(encoding="utf-8")), path
    except json.JSONDecodeError:
        return None, path


def latest_review_session_if_newer(
    previous_path: Path | None,
) -> tuple[dict[str, Any] | None, Path | None]:
    """Return the latest review session only when it differs from a prior path.
    Args:
        previous_path: Previously displayed review-session path, if any.
    Returns:
        A tuple of session payload and path when newer, otherwise None values.
    Used by:
        App Layout after handle_user_message completes a Gmail read-agent turn.
    """
    session, path = latest_review_session()
    if path is None:
        return None, None
    if previous_path and path.resolve() == previous_path.resolve():
        return None, None
    return session, path


def session_results(session: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize review-session rows for table rendering.
    Args:
        session: Review-session payload loaded from JSON.
    Returns:
        A list of normalized result dictionaries.
    Used by:
        session_counts and render_latest_session.
    """
    rows = session.get("results")
    if rows is None:
        rows = session.get("items", [])

    normalized = []
    for row in rows:
        item = dict(row)
        item["category"] = category_from_legacy(item.get("category") or item.get("classification"))
        item.setdefault("message_id", item.get("id", ""))
        normalized.append(item)
    return normalized


def session_counts(session: dict[str, Any]) -> dict[str, int]:
    """Count review-session rows by current review category.
    Args:
        session: Review-session payload loaded from JSON.
    Returns:
        A dictionary mapping category names to counts.
    Used by:
        render_latest_session when displaying metric cards.
    """
    counts = {category: 0 for category in REVIEW_CATEGORIES}
    for row in session_results(session):
        category = row.get("category")
        if category in counts:
            counts[category] += 1
    return counts


# -----------------------------------------------------------------------------
# Review Session Rendering
# -----------------------------------------------------------------------------
# This section renders the latest review metrics, filter controls, table, and download button.

def render_latest_session() -> None:
    """Render the latest review notice, metrics, table, and JSON download.
    Args:
        None.
    Returns:
        None.
    Side effects:
        Writes Streamlit UI elements to the page.
    Used by:
        App Layout after the chat interface.
    """
    session = st.session_state.latest_session
    path = st.session_state.latest_session_path
    notice = st.session_state.review_notice
    if notice:
        st.divider()
        st.info(notice)
    if not session:
        return

    rows = session_results(session)
    counts = session_counts(session)

    st.divider()
    st.subheader("Latest Review")
    cols = st.columns(5)
    cols[0].metric("Reviewed", len(rows))
    cols[1].metric("Useful", counts["useful"])
    cols[2].metric("Promotional", counts["promotional"])
    cols[3].metric("Jobs", counts["job_notifications"])
    cols[4].metric("Needs Review", counts["need_further_review"])

    if rows:
        df = pd.DataFrame(rows)
        category = st.selectbox(
            "Filter",
            ["all", *REVIEW_CATEGORIES],
            index=0,
        )
        if category != "all":
            df = df[df["category"] == category]

        visible_columns = [
            "category",
            "confidence",
            "review_source",
            "subject",
            "from",
            "reason",
            "date",
            "snippet",
            "message_id",
            "thread_id",
        ]
        st.dataframe(
            df[[column for column in visible_columns if column in df.columns]],
            use_container_width=True,
            hide_index=True,
        )

    if path and path.exists():
        st.download_button(
            "Download Session JSON",
            data=path.read_bytes(),
            file_name=path.name,
            mime="application/json",
        )


# -----------------------------------------------------------------------------
# App Layout
# -----------------------------------------------------------------------------
# This section builds the Streamlit sidebar, chat interface, progress status, and review table.

ensure_state()

st.title("Gmail Review Agent")

with st.sidebar:
    st.subheader("Chat")
    st.caption("New Chat clears the visible conversation plus chat and Gmail agent context.")
    if st.button("New Chat", use_container_width=True):
        reset_chat_state()
        st.rerun()

    st.divider()
    st.subheader("Model")
    st.session_state.model = st.text_input("Model", value=st.session_state.model)
    st.session_state.temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state.temperature),
        step=0.05,
    )
    st.caption(
        f"Context target: {CONTEXT_WINDOW_TOKENS} tokens. "
        f"Batch target: {MAX_CATEGORIZATION_BATCH} emails."
    )

    st.divider()
    st.subheader("Persistent Cache")
    cache_stats = load_cache_stats()
    st.metric("Cached emails", cache_stats["total_cached"])
    counts = cache_stats.get("category_counts", {})
    st.caption(
        "Useful: {useful} | Promotional: {promotional} | Jobs: {jobs} | Needs review: {review}".format(
            useful=counts.get("useful", 0),
            promotional=counts.get("promotional", 0),
            jobs=counts.get("job_notifications", 0),
            review=counts.get("need_further_review", 0),
        )
    )
    if cache_stats.get("updated_at"):
        st.caption(f"Updated: {cache_stats['updated_at']}")
    st.caption(f"Cache file: {REVIEW_DB_FILE.name}")

for message in st.session_state.ui_messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

prompt = st.chat_input("Ask the agent to scan Gmail or review prior results")

if prompt:
    st.session_state.ui_messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        progress_placeholder = st.empty()
        progress_lines: list[str] = []
        seen_progress_stages: set[str] = set()
        specialist_progress_started = {"value": False}

        def render_progress_line(stage: str) -> None:
            """Render one visible progress line after a specialist starts.

            Args:
                stage: Progress stage key emitted by the agent workflow.

            Returns:
                None.

            Side effects:
                Updates the assistant progress placeholder in the Streamlit chat.

            Used by:
                update_progress inside the prompt-handling UI block.
            """
            if stage in {"gmail_read_agent_at_work", "gmail_mutation_agent_at_work"}:
                specialist_progress_started["value"] = True
            if not specialist_progress_started["value"]:
                return
            message = CHAT_PROGRESS_MESSAGES.get(stage)
            if not message or stage in seen_progress_stages:
                return
            seen_progress_stages.add(stage)
            progress_lines.append(f"- {message}")
            progress_placeholder.markdown("\n".join(progress_lines))

        with st.status(PROGRESS_STAGE_LABELS["routing"], expanded=False) as status:
            try:
                def update_progress(stage: str, label: str) -> None:
                    """Update Streamlit status and append readable progress lines.

                    Args:
                        stage: Progress stage key emitted by the agent workflow.
                        label: User-facing progress label for the status widget.

                    Returns:
                        None.

                    Side effects:
                        Updates Streamlit status and progress placeholders.

                    Used by:
                        handle_user_message as its progress_callback.
                    """
                    status.update(
                        label=label or PROGRESS_STAGE_LABELS.get(stage, stage),
                        state="running",
                    )
                    render_progress_line(stage)

                previous_session_path = st.session_state.latest_session_path
                result = run_async(
                    handle_user_message(
                        prompt=prompt,
                        state=st.session_state.agent_state,
                        model=st.session_state.model,
                        temperature=st.session_state.temperature,
                        ui_messages=st.session_state.ui_messages,
                        progress_callback=update_progress,
                    )
                )
                st.session_state.agent_state = result.state
                response_text = (result.response or "").strip()
                if not response_text:
                    response_text = "I finished the run, but did not receive a text response."

                if result.route == "gmail_read_agent":
                    st.session_state.latest_session = None
                    st.session_state.latest_session_path = None
                    st.session_state.review_notice = "Gmail review is running. Waiting for a new review session..."
                    latest_session, latest_path = latest_review_session_if_newer(
                        previous_session_path
                    )
                    if latest_session is not None:
                        st.session_state.latest_session = latest_session
                        st.session_state.latest_session_path = latest_path
                        st.session_state.review_notice = None
                    else:
                        st.session_state.review_notice = (
                            "No new review table was produced for the latest Gmail turn. "
                            "The previous review table was cleared to avoid showing stale cached results."
                        )
                else:
                    st.session_state.review_notice = None

                status.update(label="Done", state="complete")
            except Exception as exc:
                response_text = f"Agent run failed: {exc}"
                status.update(label="Failed", state="error")

        response_placeholder.write(response_text)
        progress_placeholder.empty()

    st.session_state.ui_messages.append(
        {
            "role": "assistant",
            "content": response_text,
        }
    )

render_latest_session()
