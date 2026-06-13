# Claude Code Project Context

Read this file before assisting developers on this repository. It gives Claude Code the project map, safety rules, and verification habits needed to make useful changes without exposing private Gmail data.

## Project Summary

This project is a local Gmail review assistant. It combines:

- A FastMCP Gmail tool server in `server.py`.
- Gmail API operations in `gmail_client.py`.
- Google OAuth credential handling in `auth.py`.
- A LangGraph/LangChain agent workflow in `agent.py`.
- A Streamlit UI in `streamlit_app.py`.
- Smoke checks and test helpers in `diagnostics.py`, `test_auth.py`, and `test_gmail.py`.

The app can read Gmail, classify messages for review, save local review sessions, apply labels, and move selected Gmail messages to Trash.

## Architecture Flow

```text
Streamlit UI or CLI
    -> agent.py LangGraph workflow
    -> MCP client tools
    -> server.py FastMCP tools
    -> gmail_client.py Gmail API calls
    -> Gmail API
```

OAuth credentials are loaded through `auth.py`. Local review data is stored by `agent.py`.

## Main Files

| File | Role |
| --- | --- |
| `server.py` | Exposes Gmail search, read, label, preview, and trash operations as FastMCP tools. |
| `gmail_client.py` | Wraps Gmail API calls, retries, throttling, search, read, label, preview, and trash behavior. |
| `agent.py` | Defines prompts, local review tools, Gmail read/mutation agents, routing, graph nodes, and CLI flow. |
| `streamlit_app.py` | Provides the chat UI, progress messages, sidebar state, and review table controls. |
| `auth.py` | Loads, refreshes, or creates Gmail OAuth credentials. |
| `diagnostics.py` | Provides smoke checks for model, local tools, and Gmail MCP access. |
| `requirements.txt` | Installable Python package list. |
| `requirements.md` | Human-readable runtime, privacy, and setup requirements. |

## Sensitive Runtime Files

These files or directories may exist locally and can contain private data:

- `credentials.json`
- `token.json`
- `gmail_review_db.json`
- `review_sessions/`
- Gmail exports, cached summaries, notebooks, or generated review artifacts

Do not read, print, summarize, commit, or quote their contents unless the user explicitly asks and the task truly requires it. Prefer describing them generically by filename and purpose.

## Privacy Rules

Never expose:

- OAuth client secrets.
- OAuth access or refresh tokens.
- Gmail message bodies, snippets, subjects, sender addresses, or message IDs.
- Cached Gmail classifications or summaries.
- Saved review session contents.
- Personal file paths or mailbox-derived data.

When documenting setup, use placeholders and generic names only.

## Development Guardrails

- Keep changes tightly scoped to the user's objective.
- Do not rename MCP tools, local tools, public functions, return shapes, prompts, constants, or files unless explicitly approved.
- Treat Gmail mutation behavior with extra care. Labeling and Trash operations modify mailbox state.
- Do not run OAuth flows, Gmail API calls, Streamlit servers, or mutation tools without user approval.
- Preserve existing behavior during documentation, formatting, or organization tasks.
- Prefer small diffs and clear verification over broad refactors.

## Tool Safety Map

Read-oriented MCP tools:

- `search_gmail`
- `search_gmail_date_window`
- `get_gmail_message`
- `preview_delete`

Mutation-capable MCP tools:

- `apply_gmail_label`
- `trash_gmail_messages_by_label`
- `move_gmail_message_to_trash`

Local review/session tools in `agent.py` may read or write local review cache/session files. Treat those outputs as mailbox-derived data.

## Runtime Requirements

Use `requirements.txt` for package installation and `requirements.md` for full setup context.

The local model runtime defaults are defined in `agent.py`:

- Base URL: `http://127.0.0.1:1234/v1`
- API key placeholder: `lm-studio`
- Default model: `qwen/qwen3.5-9b`

The Gmail OAuth scope is defined in `auth.py` and currently uses Gmail modify access.

## Safe Verification

Safe checks that do not touch Gmail:

```bash
python -c "import importlib.util; mods=['streamlit','pandas','googleapiclient','google_auth_oauthlib','google.auth','langchain_core','langchain_openai','langchain_mcp_adapters','langgraph','mcp']; missing=[m for m in mods if importlib.util.find_spec(m) is None]; print('missing=' + ','.join(missing) if missing else 'all dependency modules found')"
```

```bash
python -m py_compile server.py gmail_client.py agent.py streamlit_app.py auth.py diagnostics.py test_auth.py test_gmail.py
```

`py_compile` may create bytecode cache files. Those are generated artifacts and should not be committed.

## Claude Code Working Notes

- Start by reading the relevant file sections, not the private runtime artifacts.
- If a task is ambiguous, ask a focused question before changing behavior.
- For code reviews, lead with bugs, regressions, risks, and missing tests.
- For implementation tasks, make the smallest useful change and then verify it.
- Keep explanations concise and grounded in file names, functions, and observed behavior.
