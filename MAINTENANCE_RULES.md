# Maintenance Rules

These rules are triggered only when the user gives the command:

```text
update_mdfiles
```

Use this maintenance flow to refresh dependency documentation, agent context files, and related Markdown files. Do not run this flow automatically after every small code change.

## Maintenance Scope

When `update_mdfiles` is requested, review and update as needed:

- `requirements.txt`
- `requirements.md`
- `CLAUDE.md`
- `AGENTS.md`
- Suggested `.gitignore` changes, with user approval before editing `.gitignore`

The goal is to keep dependency, setup, privacy, and agent-context documentation aligned with the current codebase.

## Requirements Update Rules

1. Scan Python imports across the project.
2. Separate standard-library imports from third-party dependencies.
3. Map imported modules to install package names.
4. Ensure `requirements.txt` includes all direct third-party packages needed to run the app end to end.
5. Preserve existing pinned versions unless there is a clear reason to change them.
6. When adding packages, prefer locally verified versions over guessing.
7. Do not add speculative, unused, or transitive-only dependencies unless the app directly requires them.
8. Keep `requirements.md` GitHub-friendly, clear, and privacy-safe.
9. Do not include secrets, personal data, mailbox-derived examples, or local private paths in requirements documentation.

## Agent Context Update Rules

1. Maintain `CLAUDE.md` for Claude Code context.
2. Maintain `AGENTS.md` for Codex-style agent context.
3. Keep both files privacy-safe.
4. Update these files only when architecture, entrypoints, tools, safety rules, verification workflows, or dependency requirements have materially changed.
5. Do not include private Gmail data, tokens, credentials, message IDs, sender addresses, snippets, subjects, summaries, or review session contents.
6. Keep the files concise enough for coding agents to load as project context.

## Gitignore Suggestion Rules

1. Suggest `.gitignore` updates when new generated, private, cache, diagnostics-output, or mailbox-derived files are introduced.
2. Explain why each suggested ignore pattern is needed.
3. Do not update `.gitignore` until the user approves.
4. Never use `.gitignore` as a substitute for protecting already-exposed secrets.

Suggested sensitive/generated patterns to consider when relevant:

```gitignore
credentials.json
token.json
gmail_review_db.json
review_sessions/
__pycache__/
*.pyc
```

Only add patterns that are appropriate for the current repository state.

## Privacy Review Rules

Before finishing `update_mdfiles`, scan changed Markdown files for obvious secret-like or private patterns, including:

- OAuth tokens.
- Refresh tokens.
- Client secrets.
- Private keys.
- Gmail addresses.
- Message IDs.
- Personal mailbox-derived content.

Use generic filenames and setup descriptions instead of real values.

## Verification Checklist

Use the lightest checks that match the maintenance changes:

1. Read back changed Markdown files.
2. Run `git diff --check` on changed Markdown and requirements files.
3. Run the safe dependency availability check if dependencies changed.
4. Do not start Streamlit, OAuth, Gmail, or long-running servers.
5. Do not run Gmail API calls unless the user explicitly approves.

Safe dependency availability check:

```bash
python -c "import importlib.util; mods=['streamlit','pandas','googleapiclient','google_auth_oauthlib','google.auth','langchain_core','langchain_openai','langchain_mcp_adapters','langgraph','mcp']; missing=[m for m in mods if importlib.util.find_spec(m) is None]; print('missing=' + ','.join(missing) if missing else 'all dependency modules found')"
```

## Reporting Format

When maintenance is complete, report:

1. Files changed.
2. Dependency updates made or confirmed unnecessary.
3. Agent context updates made or confirmed unnecessary.
4. `.gitignore` suggestions, if any.
5. Privacy checks performed.
6. Verification performed or skipped.
