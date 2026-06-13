# Project Requirements

This document lists the software, services, local files, and safety requirements needed to run the Gmail MCP review application end to end.

It intentionally does not include private tokens, OAuth secrets, Gmail message content, review exports, cached summaries, message IDs, or personal mailbox data.

## Python Runtime

- Python 3.11 or newer is recommended.
- A virtual environment is recommended for local development.
- Install Python packages from `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

## Python Packages

The application imports the following third-party packages.

| Package | Used For |
| --- | --- |
| `langchain-mcp-adapters` | Connecting the LangGraph agent to MCP tools. |
| `langchain-openai` | Calling the local OpenAI-compatible chat model endpoint. |
| `langchain-core` | LangChain message and tool primitives. |
| `langgraph` | Agent workflow graph, routing, and ReAct agent support. |
| `openai` | OpenAI-compatible client dependency used by LangChain. |
| `mcp` | FastMCP server used to expose Gmail tools. |
| `google-api-python-client` | Gmail API client creation and Gmail API calls. |
| `google-auth` | Google credential loading and refresh support. |
| `google-auth-oauthlib` | Local OAuth browser flow for Gmail authorization. |
| `streamlit` | Web UI for the Gmail review assistant. |
| `pandas` | Table and dataframe handling in the Streamlit UI. |

## Current `requirements.txt`

The current install file pins the app's direct runtime dependencies:

```text
# Agent and MCP runtime
langchain-core==1.3.2
langchain-mcp-adapters==0.2.2
langchain-openai==1.2.1
langgraph==1.1.9
mcp==1.27.0
openai==2.32.0

# Gmail API and OAuth
google-api-python-client==2.194.0
google-auth==2.49.2
google-auth-oauthlib==1.3.1

# Streamlit UI and data handling
pandas==3.0.2
streamlit==1.56.0
```

For a complete end-to-end environment, install the packages from `requirements.txt` in the active Python environment.

## External Services

### Gmail API

The project requires access to the Gmail API through a Google Cloud OAuth client.

Required setup:

- A Google Cloud project with Gmail API enabled.
- An OAuth client configured for a local desktop application flow.
- Gmail modify scope access, because the app can read messages, apply labels, and move messages to Trash.

### Local Chat Model Runtime

The agent is configured to call an OpenAI-compatible local model endpoint.

Default runtime expectation:

- Base URL: `http://127.0.0.1:1234/v1`
- API key placeholder: `lm-studio`
- Default model name: `qwen/qwen3.5-9b`

The local model server must be running before using the agent or Streamlit app.

## Required Local Files

These files are required or generated at runtime. They must not be committed to a public repository.

| File or Directory | Purpose | Commit Safety |
| --- | --- | --- |
| `credentials.json` | OAuth client configuration downloaded from Google Cloud. | Do not commit. |
| `token.json` | Generated OAuth user token after authorization. | Do not commit. |
| `gmail_review_db.json` | Local cache of Gmail review results. | Do not commit if it may contain mailbox-derived data. |
| `review_sessions/` | Saved review session files. | Do not commit if it may contain mailbox-derived data. |

## Privacy And Security Requirements

Never document, commit, or share:

- OAuth client secrets.
- OAuth access or refresh tokens.
- Gmail message bodies, snippets, subjects, sender addresses, or message IDs.
- Generated Gmail review summaries.
- Saved review sessions.
- Local cache files containing mailbox-derived data.
- Personal file paths that reveal private user information.

Use generic filenames and setup instructions when documenting sensitive resources.

## Safe Setup Checklist

1. Create and activate a Python virtual environment.
2. Install packages with `python -m pip install -r requirements.txt`.
3. Confirm any additional packages from this document are installed in the same environment.
4. Place `credentials.json` in the project root.
5. Start the local OpenAI-compatible model server.
6. Run the app entrypoint only when ready to authorize Gmail access.

## Verification Commands

Use this import check to confirm the main third-party dependencies are available:

```bash
python -c "import streamlit, pandas, googleapiclient, google_auth_oauthlib, google.auth, langchain_core, langchain_openai, langchain_mcp_adapters, langgraph, mcp"
```

Use this syntax check after code changes:

```bash
python -m py_compile server.py gmail_client.py agent.py streamlit_app.py auth.py diagnostics.py test_auth.py test_gmail.py
```

The syntax check may create Python bytecode cache files locally. Those files are generated artifacts and should not be committed.

## Application Entry Points

| File | Purpose |
| --- | --- |
| `server.py` | Starts the FastMCP Gmail tool server. |
| `agent.py` | Runs the LangGraph Gmail review assistant and CLI workflow. |
| `streamlit_app.py` | Runs the Streamlit chat and review UI. |
| `auth.py` | Handles Gmail OAuth credential creation and refresh. |
| `diagnostics.py` | Runs local smoke checks for model, tools, and Gmail MCP access. |

## Gmail Safety Notes

Some tools can modify Gmail state by applying labels or moving messages to Trash. Review tool responses carefully before running mutation workflows, and keep OAuth credentials private.
