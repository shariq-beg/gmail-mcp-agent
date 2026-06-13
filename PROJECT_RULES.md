# Project Rules

These are the always-on rules for future development sessions in this Gmail MCP project. Load and follow these rules whenever enhancing, modifying, documenting, or reviewing the codebase.

## Project Safety Rules

1. Preserve existing tested functionality unless the user explicitly approves behavior changes.
2. Do not rewrite working logic during documentation, formatting, cleanup, or organization tasks.
3. Keep edits scoped to the current user-approved objective.
4. Do not rename public functions, MCP tools, local tools, constants, prompts, files, return keys, or CLI behavior unless explicitly approved.
5. Do not change Gmail OAuth scopes unless explicitly approved.
6. Treat Gmail mutation paths as high-risk.

## Privacy Rules

1. Never document, print, summarize, or commit private Gmail-derived data.
2. Do not expose OAuth tokens, refresh tokens, client secrets, message IDs, sender addresses, subjects, snippets, summaries, cached reviews, or review session contents.
3. Refer to sensitive runtime files only generically by filename and purpose.
4. Sensitive files include:
   - `credentials.json`
   - `token.json`
   - `gmail_review_db.json`
   - `review_sessions/`
   - notebooks or generated files containing mailbox data
5. Do not inspect sensitive files unless the user explicitly asks and the task truly requires it.

## Code Organization Rules

1. Keep the existing file/module layout unless the user approves splitting files.
2. Use clear section headers in Python files.
3. Section headers should include a short one-line description.
4. Prefer adding navigation structure in place over moving large blocks.
5. Avoid import movement unless required and behavior-neutral.

## Docstring Rules

1. Every function should have a compact docstring.
2. Docstrings should describe:
   - What the function does
   - Args
   - Returns
   - Raises, if applicable
   - Side effects, if applicable
   - Used by
3. Avoid excessive blank lines inside docstrings.
4. Tool docstrings must be action-oriented, clear, and safe for agent/tool use.
5. Tool docstrings should clarify whether the tool reads Gmail or mutates Gmail state.

## Gmail Tool Safety Rules

Read-oriented MCP tools:

- `search_gmail`
- `search_gmail_date_window`
- `get_gmail_message`
- `preview_delete`

Mutation-capable MCP tools:

- `apply_gmail_label`
- `trash_gmail_messages_by_label`
- `move_gmail_message_to_trash`

Rules:

1. Do not run Gmail API calls without user approval.
2. Do not run mutation tools without explicit user approval.
3. Preserve confirmation and preview behavior around destructive flows.
4. Be extra careful with label and Trash operations.

## User Message Interpretation Rules

1. Avoid brittle word parsing unless absolutely necessary.
2. Prefer LLM interpretation of user messages over keyword matching.
3. Use structured prompts, tool descriptions, and model reasoning for intent classification where possible.
4. Use explicit parsing only for constrained formats such as dates, file paths, CLI flags, JSON, or confirmed command syntax.
5. If user intent is ambiguous and the result could affect Gmail state, privacy, architecture, or data loss, ask a clarifying question.

## Verification Rules

1. Use the lightest verification that matches the change.
2. Always place verification scripts under `diagnostics/` or another user-approved diagnostics area.
3. Never add ad hoc testing or verification functions to main application modules.
4. Do not include testing-only functions in production runtime files such as `agent.py`, `server.py`, `gmail_client.py`, `streamlit_app.py`, or `auth.py`.
5. Tell the user before running checks that create bytecode cache.
6. Do not start Streamlit, OAuth, Gmail, or long-running servers without approval.
7. If verification is skipped, say so clearly and explain why.
8. Suggest `.gitignore` updates when new generated, private, cache, or diagnostics-output files are introduced.
9. Update `.gitignore` only after user approval.

## Maintenance Reminder Rule

1. If `MAINTENANCE_RULES.md`, `requirements.md`, `CLAUDE.md`, or `AGENTS.md` have not been reviewed or updated for more than 7 days, briefly remind the user that maintenance may be due.
2. Do not run maintenance automatically.
3. Recommend the command `update_mdfiles` when maintenance is due.
4. Do not repeat the reminder within the same session unless the user asks.

Safe dependency availability check:

```bash
python -c "import importlib.util; mods=['streamlit','pandas','googleapiclient','google_auth_oauthlib','google.auth','langchain_core','langchain_openai','langchain_mcp_adapters','langgraph','mcp']; missing=[m for m in mods if importlib.util.find_spec(m) is None]; print('missing=' + ','.join(missing) if missing else 'all dependency modules found')"
```

Safe syntax check:

```bash
python -m py_compile server.py gmail_client.py agent.py streamlit_app.py auth.py diagnostics.py test_auth.py test_gmail.py
```

## Git And File Handling Rules

1. Never revert user changes unless explicitly requested.
2. Be careful in dirty working trees.
3. Do not delete generated files without checking the path and getting approval when needed.
4. Use small, readable diffs.
5. Keep generated/private artifacts out of documentation and commits.

## Communication Rules

1. Explain what will be changed before editing.
2. Report what changed and what was verified.
3. Keep final summaries concise.
4. Surface risks, skipped checks, and assumptions clearly.
5. When reviewing code, prioritize bugs, regressions, safety risks, and missing tests.
