# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
# This section imports standard library, Google API, and auth helpers used by Gmail operations.

from __future__ import annotations

import base64
import email.utils
import json
import random
import time
from typing import Any

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from auth import get_credentials


# -----------------------------------------------------------------------------
# Gmail Client Configuration
# -----------------------------------------------------------------------------
# This section defines Gmail API constants, retry policy, and shared client state.

_USER_ID = "me"
_PAGE_SIZE = 500
_MIN_REQUEST_INTERVAL_SECONDS = 0.05
_MAX_RETRIES = 5
_MAX_BACKOFF_SECONDS = 16.0
_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}
_RETRYABLE_403_REASONS = {
    "backendError",
    "internalError",
    "rateLimitExceeded",
    "userRateLimitExceeded",
}
_SERVICE = None
_LAST_REQUEST_MONOTONIC = 0.0


# -----------------------------------------------------------------------------
# Gmail Service Setup
# -----------------------------------------------------------------------------
# This section creates and reuses the authenticated Gmail API service object.

def get_service():
    """Create or return the cached authenticated Gmail API service.
    Args:
        None.
    Returns:
        A Gmail API service object for the configured user.
    Side effects:
        May trigger credential loading or OAuth refresh through get_credentials.
    Used by:
        Gmail client operations that call the Gmail API.
    """
    global _SERVICE
    if _SERVICE is None:
        creds = get_credentials()
        _SERVICE = build("gmail", "v1", credentials=creds, cache_discovery=False)
    return _SERVICE



# -----------------------------------------------------------------------------
# Request Execution Helpers
# -----------------------------------------------------------------------------
# This section throttles Gmail API calls and retries transient API failures.

def _throttle() -> None:
    """Pause briefly when needed to keep Gmail API requests spaced out.
    Args:
        None.
    Returns:
        None.
    Side effects:
        May sleep and updates the module-level last-request timestamp.
    Used by:
        _execute before each Gmail API request.
    """
    global _LAST_REQUEST_MONOTONIC

    elapsed = time.monotonic() - _LAST_REQUEST_MONOTONIC
    remaining = _MIN_REQUEST_INTERVAL_SECONDS - elapsed
    if remaining > 0:
        time.sleep(remaining)
    _LAST_REQUEST_MONOTONIC = time.monotonic()



def _extract_error_reason(exc: HttpError) -> str | None:
    """Extract a Google API error reason from an HttpError payload.
    Args:
        exc: HttpError raised by the Google API client.
    Returns:
        The parsed error reason or status string, or None when unavailable.
    Used by:
        _is_retryable_error when deciding whether a 403 error can be retried.
    """
    content = getattr(exc, "content", None)
    if not content:
        return None

    try:
        payload = json.loads(content.decode("utf-8"))
    except (AttributeError, UnicodeDecodeError, json.JSONDecodeError):
        return None

    error = payload.get("error", {})
    errors = error.get("errors", [])
    if errors and isinstance(errors[0], dict):
        reason = errors[0].get("reason")
        if isinstance(reason, str):
            return reason

    reason = error.get("status")
    return reason if isinstance(reason, str) else None



def _is_retryable_error(exc: HttpError) -> bool:
    """Decide whether a Gmail API HttpError should be retried.
    Args:
        exc: HttpError raised by a Gmail API request.
    Returns:
        True when the status or reason is considered transient; otherwise False.
    Used by:
        _execute during retry handling.
    """
    status = getattr(exc.resp, "status", None)
    if status in _RETRYABLE_STATUSES:
        return True
    if status != 403:
        return False
    return _extract_error_reason(exc) in _RETRYABLE_403_REASONS



def _execute(request: Any) -> dict[str, Any]:
    """Execute a Gmail API request with throttling and retry backoff.
    Args:
        request: Google API request object with an execute method.
    Returns:
        The Gmail API response dictionary.
    Raises:
        HttpError: Re-raised when the error is not retryable or retries are exhausted.
        RuntimeError: Raised if retry flow exits unexpectedly.
    Used by:
        All Gmail API operations in this module.
    """
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            _throttle()
            return request.execute()
        except HttpError as exc:
            if not _is_retryable_error(exc) or attempt == _MAX_RETRIES:
                raise
            backoff_ceiling = min(_MAX_BACKOFF_SECONDS, 2**attempt)
            time.sleep(random.uniform(0.0, backoff_ceiling))

    raise RuntimeError("Gmail request retries exhausted")


# -----------------------------------------------------------------------------
# Message Parsing Helpers
# -----------------------------------------------------------------------------
# This section extracts headers, sender domains, and text bodies from Gmail payloads.

def _get_header(headers: list[dict[str, str]], name: str) -> str:
    """Return a case-insensitive Gmail message header value.
    Args:
        headers: Gmail payload header dictionaries.
        name: Header name to search for.
    Returns:
        The header value, or an empty string when missing.
    Used by:
        read_email when normalizing Gmail message metadata.
    """
    return next(
        (h.get("value", "") for h in headers if h.get("name", "").lower() == name),
        "",
    )


def _extract_sender_domain(sender: str) -> str:
    """Extract the lowercase domain from an email sender string.
    Args:
        sender: Raw From header value.
    Returns:
        Sender domain, or an empty string when no email address is present.
    Used by:
        read_email when adding sender_domain metadata.
    """
    email_address = email.utils.parseaddr(sender)[1]
    if "@" not in email_address:
        return ""
    return email_address.rsplit("@", 1)[1].lower()


def _decode_body_data(data: str | None) -> str:
    """Decode URL-safe base64 Gmail body data into text.
    Args:
        data: Encoded body data from a Gmail payload part.
    Returns:
        Decoded UTF-8 text, or an empty string when data is missing or invalid.
    Used by:
        _extract_text_parts when reading plain-text message bodies.
    """
    if not data:
        return ""

    try:
        decoded = base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))
    except (base64.binascii.Error, ValueError):
        return ""

    return decoded.decode("utf-8", errors="replace").strip()


def _extract_text_parts(payload: dict[str, Any]) -> list[str]:
    """Recursively extract plain-text body parts from a Gmail payload.
    Args:
        payload: Gmail message payload or nested MIME part.
    Returns:
        A list of decoded plain-text body strings.
    Used by:
        read_email when building the body_text field.
    """
    mime_type = payload.get("mimeType", "")
    body_data = payload.get("body", {}).get("data")
    parts = payload.get("parts", [])

    if mime_type == "text/plain":
        text = _decode_body_data(body_data)
        return [text] if text else []

    text_parts = []
    for part in parts:
        text_parts.extend(_extract_text_parts(part))
    return text_parts


# -----------------------------------------------------------------------------
# Gmail Search Operations
# -----------------------------------------------------------------------------
# This section builds Gmail queries and retrieves message references from Gmail.

def build_date_window_query(
    base_query: str = "in:inbox",
    after: str | None = None,
    before: str | None = None,
) -> str:
    """Build a Gmail query with optional after and before date filters.
    Args:
        base_query: Base Gmail query to combine with date filters.
        after: Optional Gmail after date in YYYY/MM/DD format.
        before: Optional Gmail before date in YYYY/MM/DD format.
    Returns:
        A Gmail query string containing the base query and any date filters.
    Used by:
        search_emails_by_date_window and date-window MCP search tools.
    """
    query_parts = [part for part in [base_query.strip(), after, before] if part]
    if after:
        query_parts[-2 if before else -1] = f"after:{after}"
    if before:
        query_parts[-1] = f"before:{before}"
    return " ".join(query_parts).strip()


def search_emails_by_date_window(
    base_query: str = "in:inbox",
    after: str | None = None,
    before: str | None = None,
    max_results: int = 50,
):
    """Search Gmail using a base query plus optional date window.
    Args:
        base_query: Gmail query to combine with date filters.
        after: Optional Gmail after date in YYYY/MM/DD format.
        before: Optional Gmail before date in YYYY/MM/DD format.
        max_results: Maximum number of message references to return.
    Returns:
        A dictionary with the final query and matching message references.
    Used by:
        server.search_gmail_date_window.
    """
    query = build_date_window_query(
        base_query=base_query,
        after=after,
        before=before,
    )
    return {
        "query": query,
        "messages": search_emails(query=query, max_results=max_results),
    }



def search_emails(query: str, max_results: int = 50):
    """Search Gmail and return message references for a query.
    Args:
        query: Gmail search query string.
        max_results: Maximum number of message references to return.
    Returns:
        A list of Gmail message reference dictionaries.
    Used by:
        server.search_gmail, search_emails_by_date_window, and preview_delete_candidates.
    """
    service = get_service()
    messages = []
    page_token = None

    while len(messages) < max_results:
        page_size = min(_PAGE_SIZE, max_results - len(messages))
        result = _execute(
            service.users()
            .messages()
            .list(
                userId=_USER_ID,
                q=query,
                maxResults=page_size,
                pageToken=page_token,
                includeSpamTrash=False,
            )
        )
        messages.extend(result.get("messages", []))
        page_token = result.get("nextPageToken")
        if not page_token:
            break

    return messages


# -----------------------------------------------------------------------------
# Gmail Label Operations
# -----------------------------------------------------------------------------
# This section lists, finds, creates, and applies Gmail labels to messages.

def list_labels() -> list[dict[str, Any]]:
    """List labels available in the current Gmail account.
    Args:
        None.
    Returns:
        A list of Gmail label dictionaries.
    Used by:
        _find_label when resolving labels by name or ID.
    """
    service = get_service()
    result = _execute(service.users().labels().list(userId=_USER_ID))
    return result.get("labels", [])


def _find_label(label_name: str) -> dict[str, Any] | None:
    """Find a Gmail label by case-insensitive name or ID.
    Args:
        label_name: Gmail label display name or label ID to locate.
    Returns:
        Matching Gmail label dictionary, or None when not found.
    Used by:
        get_or_create_label and conflict recovery after label creation.
    """
    normalized = label_name.casefold()
    for label in list_labels():
        if label.get("id", "").casefold() == normalized:
            return label
        if label.get("name", "").casefold() == normalized:
            return label
    return None


def get_or_create_label(label_name: str, create_if_missing: bool = True) -> dict[str, Any]:
    """Resolve a Gmail label, optionally creating it when missing.
    Args:
        label_name: Gmail label display name or label ID.
        create_if_missing: Whether to create the label if it does not already exist.
    Returns:
        A Gmail label dictionary.
    Raises:
        ValueError: Raised when the label is missing and creation is disabled.
        HttpError: Re-raised for Gmail API label creation failures.
    Side effects:
        May create a new Gmail label.
    Used by:
        apply_label_to_email and search_emails_by_label.
    """
    label = _find_label(label_name)
    if label:
        return label

    if not create_if_missing:
        raise ValueError(f"Gmail label not found: {label_name}")

    service = get_service()
    try:
        return _execute(
            service.users()
            .labels()
            .create(
                userId=_USER_ID,
                body={
                    "name": label_name,
                    "labelListVisibility": "labelShow",
                    "messageListVisibility": "show",
                },
            )
        )
    except HttpError as exc:
        if getattr(exc.resp, "status", None) == 409:
            label = _find_label(label_name)
            if label:
                return label
        raise


def apply_label_to_email(
    message_id: str,
    label_name: str,
    create_if_missing: bool = True,
) -> dict[str, Any]:
    """Apply a Gmail label to one message and return labeling details.
    Args:
        message_id: Gmail message ID to label.
        label_name: Gmail label name or ID to apply.
        create_if_missing: Whether to create the label if it does not exist.
    Returns:
        A dictionary describing the message, label, and resulting label IDs.
    Side effects:
        Modifies the selected Gmail message by adding the resolved label.
    Used by:
        server.apply_gmail_label and confirmed Gmail mutation flows.
    """
    email = read_email(message_id)
    label = get_or_create_label(
        label_name=label_name,
        create_if_missing=create_if_missing,
    )
    service = get_service()
    result = _execute(
        service.users()
        .messages()
        .modify(
            userId=_USER_ID,
            id=message_id,
            body={"addLabelIds": [label["id"]]},
        )
    )

    return {
        "id": result.get("id"),
        "thread_id": email.get("thread_id"),
        "subject": email.get("subject", ""),
        "from": email.get("from", ""),
        "label_name": label.get("name", label_name),
        "label_id": label.get("id"),
        "label_ids": result.get("labelIds", []),
    }


def search_emails_by_label(label_name: str, max_results: int = 10):
    """Search Gmail for messages that have a specific label.
    Args:
        label_name: Gmail label display name or label ID to search.
        max_results: Maximum number of message references to return.
    Returns:
        A dictionary with label metadata and matching message references.
    Raises:
        ValueError: Raised when the label cannot be found.
    Used by:
        trash_emails_by_label.
    """
    label = get_or_create_label(label_name=label_name, create_if_missing=False)
    service = get_service()
    messages = []
    page_token = None

    while len(messages) < max_results:
        page_size = min(_PAGE_SIZE, max_results - len(messages))
        result = _execute(
            service.users()
            .messages()
            .list(
                userId=_USER_ID,
                labelIds=[label["id"]],
                maxResults=page_size,
                pageToken=page_token,
                includeSpamTrash=False,
            )
        )
        messages.extend(result.get("messages", []))
        page_token = result.get("nextPageToken")
        if not page_token:
            break

    return {
        "label_name": label.get("name", label_name),
        "label_id": label.get("id"),
        "messages": messages,
    }


def trash_emails_by_label(label_name: str, max_results: int = 10) -> dict[str, Any]:
    """Move messages with a specific Gmail label to Trash.
    Args:
        label_name: Gmail label display name or label ID to search.
        max_results: Maximum number of labeled messages to move to Trash.
    Returns:
        A dictionary with label metadata, trash count, and per-message results.
    Side effects:
        Moves each selected Gmail message to Trash.
    Used by:
        server.trash_gmail_messages_by_label.
    """
    search_result = search_emails_by_label(
        label_name=label_name,
        max_results=max_results,
    )
    trashed = []
    for ref in search_result["messages"]:
        trashed.append(trash_email(ref["id"]))

    return {
        "label_name": search_result["label_name"],
        "label_id": search_result["label_id"],
        "requested_max": max_results,
        "trashed_count": len(trashed),
        "trashed": trashed,
    }


# -----------------------------------------------------------------------------
# Gmail Message Read Operations
# -----------------------------------------------------------------------------
# This section reads full Gmail messages and normalizes fields for agents and tools.


def read_email(message_id: str):
    """Read and normalize a Gmail message by message ID.
    Args:
        message_id: Gmail message ID to fetch.
    Returns:
        A dictionary with message IDs, headers, snippet, body text, labels, and metadata.
    Used by:
        server.get_gmail_message, apply_label_to_email, preview_delete_candidates, and trash_email.
    """
    service = get_service()
    msg = _execute(
        service.users()
        .messages()
        .get(userId=_USER_ID, id=message_id, format="full")
    )

    payload = msg.get("payload", {})
    headers = payload.get("headers", [])
    subject = _get_header(headers, "subject")
    sender = _get_header(headers, "from")
    sender_domain = _extract_sender_domain(sender)
    recipient = _get_header(headers, "to")
    date = _get_header(headers, "date")
    list_unsubscribe = _get_header(headers, "list-unsubscribe")
    snippet = msg.get("snippet", "")
    body_text = "\n\n".join(_extract_text_parts(payload))
    label_ids = msg.get("labelIds", [])

    return {
        "id": msg["id"],
        "thread_id": msg.get("threadId"),
        "date": date,
        "subject": subject,
        "from": sender,
        "sender_domain": sender_domain,
        "to": recipient,
        "snippet": snippet,
        "body_text": body_text,
        "label_ids": label_ids,
        "gmail_labels": label_ids,
        "list_unsubscribe_present": bool(list_unsubscribe),
    }


# -----------------------------------------------------------------------------
# Gmail Trash Preview And Mutation Operations
# -----------------------------------------------------------------------------
# This section previews delete candidates and moves selected Gmail messages to Trash.


def preview_delete_candidates(query: str, max_results: int = 10):
    """Return lightweight previews for emails that match a delete query.
    Args:
        query: Gmail search query used to find candidate messages.
        max_results: Maximum number of candidate previews to return.
    Returns:
        A list of lightweight message preview dictionaries.
    Used by:
        server.preview_delete before any Gmail trash action is taken.
    """

    previews = []
    for ref in search_emails(query=query, max_results=max_results):
        email = read_email(ref["id"])
        previews.append(
            {
                "id": email["id"],
                "thread_id": email["thread_id"],
                "subject": email["subject"],
                "from": email["from"],
                "snippet": email["snippet"],
                "label_ids": email["label_ids"],
            }
        )
    return previews



def trash_email(message_id: str):
    """Move one Gmail message to Trash and return a summary.
    Args:
        message_id: Gmail message ID to move to Trash.
    Returns:
        A dictionary describing the trashed message and resulting label IDs.
    Side effects:
        Moves the selected Gmail message to Trash.
    Used by:
        server.move_gmail_message_to_trash and trash_emails_by_label.
    """
    email = read_email(message_id)
    service = get_service()
    result = _execute(
        service.users()
        .messages()
        .trash(userId=_USER_ID, id=message_id)
    )

    return {
        "id": result.get("id"),
        "thread_id": email.get("thread_id"),
        "subject": email.get("subject", ""),
        "from": email.get("from", ""),
        "label_ids": result.get("labelIds", []),
    }
