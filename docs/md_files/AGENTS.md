# Codex Project Context

This file gives Codex-style coding agents the working context for this repository. Read it before making edits, running commands, or proposing architecture changes.

## What This Project Is

This repository contains a local Gmail review assistant. It uses a LangGraph/LangChain agent to route user requests, call Gmail MCP tools, classify email review results, and support a Streamlit chat UI.

The app can perform both read-only Gmail operations and Gmail mutations such as applying labels or moving messages to Trash. Treat mutation paths as high-risk.

## Repo Map

| Path | Purpose |
| --- | --- |
| `agent.py` | Main agent module: prompts, config, local review tools, tool loading, routing, graph nodes, workflow execution, and CLI entrypoint. |
| `server.py` | FastMCP server exposing Gmail operations as MCP tools. |
| `gmail_client.py` | Gmail API client layer with service creation, throttling, retries, search, read, label, preview, and trash functions. |
| `auth.py` | Google OAuth credential loading, refresh, and local authorization flow. |
| `streamlit_app.py` | Streamlit UI, session state, progress display, chat rendering, and review table interactions. |
| `diagnostics.py` | Smoke-test helpers for LLM, tools, and Gmail MCP connectivity. |
| `test_auth.py` | Simple credential smoke check. |
| `test_gmail.py` | MCP client smoke check against the Gmail MCP server. |
| `requirements.txt` | Python packages required by the app. |
| `requirements.md` | Human-facing runtime, setup, and privacy requirements. |

## Runtime Flow

```text
streamlit_app.py or agent.py CLI
    -> agent.py route_gmail_request
    -> Gmail read or mutation workflow node
    -> LangGraph/ReAct agent
    -> langchain_mcp_adapters MCP client
    -> server.py FastMCP tool
    -> gmail_client.py
    -> Gmail API
```

Local review cache/session helpers in `agent.py` can also read and write project-local JSON review data.

## Important Project Assumptions

- The default chat model endpoint is OpenAI-compatible and local.
- The default model configuration lives in `agent.py`.
- Gmail OAuth uses the Gmail modify scope in `auth.py`.
- The project is intended to run locally, not as a hosted multi-user service.
- Existing Gmail behavior is assumed tested; avoid changing it during docs or cleanup tasks.

## Sensitive Files And Data

Do not inspect, print, summarize, or commit sensitive local data unless the user explicitly asks and the task requires it.

Sensitive examples:

- `credentials.json`
- `token.json`
- `gmail_review_db.json`
- `review_sessions/`
- Gmail exports
- Cached email summaries
- Notebook outputs containing mailbox data
- Any token, credential, message body, sender, subject, snippet, or message ID

Documentation should refer to these generically and never include real values.

## Editing Rules For Codex

- Prefer minimal, behavior-preserving edits.
- Do not rewrite working logic during organization, documentation, or dependency tasks.
- Do not rename public functions, MCP tools, local tools, return keys, constants, prompts, files, or CLI behavior unless the user approves it.
- Do not change OAuth scopes casually.
- Do not broaden Gmail mutation behavior.
- Keep Gmail Trash and label operations visibly confirmable in user-facing flows.
- Respect dirty working trees and never revert user changes without explicit instruction.

## MCP Tool Safety

Read-oriented server tools:

- `search_gmail`
- `search_gmail_date_window`
- `get_gmail_message`
- `preview_delete`

Mutation-capable server tools:

- `apply_gmail_label`
- `trash_gmail_messages_by_label`
- `move_gmail_message_to_trash`

When changing mutation-capable tools, verify argument handling, return shape, and user confirmation behavior.

## Agent Areas In `agent.py`

Key areas to inspect before editing agent behavior:

- Runtime constants and category definitions.
- Router and agent prompts.
- Review cache/session helpers.
- Local review tools.
- Gmail read-agent tools.
- Gmail mutation-agent tools.
- Workflow graph nodes.
- `build_workflow_graph`.
- `handle_user_message`.
- CLI helpers and entrypoint.

## Safe Commands

Use these checks when appropriate and with user approval if bytecode cache generation matters:

```bash
python -m py_compile server.py gmail_client.py agent.py streamlit_app.py auth.py diagnostics.py test_auth.py test_gmail.py
```

Use this dependency availability check without importing the full app stack:

```bash
python -c "import importlib.util; mods=['streamlit','pandas','googleapiclient','google_auth_oauthlib','google.auth','langchain_core','langchain_openai','langchain_mcp_adapters','langgraph','mcp']; missing=[m for m in mods if importlib.util.find_spec(m) is None]; print('missing=' + ','.join(missing) if missing else 'all dependency modules found')"
```

Avoid running commands that:

- Start OAuth flows.
- Read live Gmail.
- Mutate Gmail.
- Start long-running servers.
- Print private local review data.

Run those only when the user explicitly authorizes them.

## Dependency Notes

Install runtime dependencies from:

```bash
python -m pip install -r requirements.txt
```

See `requirements.md` for the full setup and privacy checklist.

## Response Style For Future Agents

- Be explicit about what changed and what was verified.
- Report skipped verification honestly.
- Ask before touching sensitive files or running live Gmail flows.
- Keep documentation changes privacy-safe.
- Keep code changes small, readable, and consistent with existing project structure.
