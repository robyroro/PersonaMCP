# Contributing

Use synthetic data in issues, fixtures, and pull requests. Never attach personal exports,
screenshots of private messages, databases, generated profiles, or model caches.

```sh
git clone https://github.com/robyroro/PersonaMCP.git
cd PersonaMCP
uv sync --locked
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy src
uv run pytest
```

Semantic development is optional: `uv sync --locked --extra semantic`. Model preparation is an
explicit download; ordinary tests use a clearly named vector fixture and need no model or network.
Real model checks should use temporary local data directories and must not publish their examples.

Adapters implement `Importer.parse(text, source)`, returning normalized conversations/messages.
Keep date assumptions explicit, distinguish system/media records, preserve incoming context,
and add synthetic regressions for pagination, duplicate imports, malformed input, and identity.
Unsupported variants should fail clearly rather than silently invent a successful import.

Schema migrations belong in storage and advance `PRAGMA user_version`. Never open a newer database
with an older schema. Changes to identity, text, conversations, or privacy operations must account
for derived interactions, FTS, profiles, and local vectors. Retrieval additions must prove that an
exact recipient filter never falls back to someone else's conversations.

Keep changes focused. Include the problem, behavior change, and validation in the pull request.
For security issues, use the private process in [SECURITY.md](SECURITY.md).
