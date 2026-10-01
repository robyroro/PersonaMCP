# MVP implementation plan

The workspace initially contains personal Instagram HTML and Snapchat ZIP exports, no source
repository. These files stay untouched and ignored. Public fixtures will be entirely synthetic.

1. Normalize platform adapters into typed conversations and messages. Require an explicit owner
   name/alias per platform; never infer identity from frequency. Support JSON, JSONL, WhatsApp TXT,
   Instagram JSON/HTML, and Snapchat JSON/HTML inside ZIPs without extracting archives.
2. Store in versioned SQLite, with foreign keys, transactions, stable IDs, and FTS5. Rebuild
   interactions across pagination boundaries; collect short reply bursts with bounded preceding
   incoming context. Changing identity recomputes flags and invalidates derived data.
3. Compute deterministic global, recipient, and explicitly assigned context profiles. Only owner
   text feeds style analysis. Phrase thresholds distinguish repeated evidence from single mistakes.
4. Implement bounded lexical and optional local multilingual sentence-transformer retrieval.
   Store float32 vectors in SQLite. Download models only through an explicit preparation command;
   runtime model loading is offline with remote code disabled.
5. Share one service between Typer CLI and official MCP Python SDK stdio tools. Tools disclose
   bounded untrusted quoted examples; a recipient filter never silently falls back to other people.
6. Add temporal holdout retrieval benchmarks, synthetic parser/privacy/regression tests, real
   export smoke checks, packaging, locked dependencies, CI, and usage/security documentation.

## Decisions and limits

- Python 3.11+, SDK v1 stable release line; no HTTP listener, web UI, cloud provider, or reply generator.
- Plain SQLite is local but not encrypted; OS account controls and disk encryption are recommended.
- Exact cosine scans are reasonable for an MVP; future indexes can implement the same provider
  boundary. Explicit indexing reports progress and can resume after interruption.
- Recipient names are platform-scoped. Group examples are conservatively excluded from
  person-specific retrieval; group identity and psychological relationships are not inferred.
- Context labels are user-assigned, never guessed from names or presumed relationships.
- Exports vary over time. Unsupported records must fail clearly or be counted as omitted.
- Privacy deletions invalidate profiles and vectors, purge FTS, and vacuum the local database.
- Benchmarks compare a retrieved historical response with a hidden later response. They measure
  retrieval and surface style similarity, not authorship, psychological identity, or generation quality.
