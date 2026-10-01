# PersonaMCP

**Your communication style and memory for AI agents.**

PersonaMCP imports your conversation exports, measures how **you** write, and gives an AI agent
a compact style profile plus relevant incoming-message/reply examples. Your raw history stays
on your computer. There is no model training, web dashboard, cloud embedding requirement,
telemetry, or automatic reply generation.

Writing preferences are usually too vague: “sound casual” does not capture someone who uses
lowercase, sends three short messages, switches languages, or writes differently to a colleague.
PersonaMCP gives the connected writer evidence instead of a guessed personality.

## Install

Python 3.11 or newer. Install from this repository; a PyPI release is not currently published.

```sh
git clone https://github.com/robyroro/PersonaMCP.git
cd PersonaMCP
uv sync --locked
uv run persona --help
```

Or install into your own virtual environment:

```sh
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install .
persona --help
```

In a uv checkout, prefix the `persona` commands below with `uv run`. With an activated pip
installation, run `persona` directly.

The base installation supports import, analysis, FTS search, lexical interaction retrieval, and
MCP. Semantic retrieval is an optional, fully local extra:

After initialization and imports (described below):

```sh
uv sync --locked --extra semantic
uv run persona model prepare
uv run persona index
```

With pip, use `python -m pip install '.[semantic]'`. Preparing the model explicitly downloads
weights from Hugging Face and pins their immutable revision. It does **not** read or send chats.
Indexing and subsequent queries load only local files with remote code disabled.

## Quick start

```sh
persona init
persona config set-name "Robert"
persona config add-alias "roby"
persona config set-name "Exact Instagram display name" --platform instagram
persona config set-name "snapchat_username" --platform snapchat

persona import instagram ./instagram-export/
persona import snapchat ./snapchat-export/
persona import whatsapp ./chat.txt
persona import json ./messages.json

persona stats
persona analyze
persona search "cat costa"
persona similar "mai vii azi?" --person "David"
persona writing-context "ce faci diseara?" --person "David" --platform instagram
```

For a synthetic first run, use a separate data directory:

```sh
persona --home ./sample-persona init
persona --home ./sample-persona config set-name "Owner"
persona --home ./sample-persona import json ./examples/messages.json
persona --home ./sample-persona analyze
persona --home ./sample-persona writing-context "mai vii azi la cafea?" --person "Alex"
```

Put private imports and custom data directories **outside your repository**. The included
`.gitignore` covers conventional private folders but cannot protect every arbitrarily named path.

Identity must match an exact exported sender name or sender ID. Platform aliases override global
names on that platform. No participant is guessed from message volume. An import matching no
owner messages fails before writing anything. Changing names/aliases recomputes ownership and
interactions and invalidates profiles/vectors; run `analyze` and `index` again.

## Supported imports

| Platform | Supported input | Notes |
| --- | --- | --- |
| Instagram | `message_N.json` or current Meta `message_N.html` folders | Pagination is merged. Common broken JSON Unicode is repaired. HTML scripts/links/media are never executed. |
| Snapchat | `chat_history.json`, supported chat-card HTML subpages, or original ZIP exports | Direct recipient-keyed and nested JSON layouts. ZIPs are read without extraction; JSON takes precedence over duplicate HTML. |
| WhatsApp | UTF-8 TXT, Android and bracketed iOS timestamps | Slash-separated dates; day/month default, `--month-first` for US exports. Multiline text is retained; system/media notices are excluded from style. |
| Generic | JSON conversation object, `conversations` object, message array, or JSONL | See the schema below. |

Only chat files are imported. Media is not opened, transcribed, downloaded, or analyzed. Unsupported
variants fail clearly. A malformed file rolls back the import as a whole. Original files stay
untouched. Content hashes and stable message IDs prevent repeated imports from duplicating rows.

WhatsApp has no stable export thread ID. By default the filename identifies the conversation;
use `--conversation-id "stable-chat-name"` when importing renamed or refreshed exports.
Offsetless timestamps use a documented UTC convention for wall-clock ordering; they are not
claimed to have been recorded in UTC. Instagram/Snapchat HTML support English export dates.

Generic JSON:

```json
{
  "id": "stable-conversation-id",
  "platform": "json",
  "title": "Alex",
  "participants": ["Owner", "Alex"],
  "context": "casual",
  "messages": [
    {"id": "external-message-id", "sender": "Alex", "sender_id": "account-123",
     "text": "mai vii azi?", "timestamp": "2024-01-01T10:00:00Z"},
    {"sender": "Owner", "text": "da gen vin acu", "timestamp": "2024-01-01T10:00:10Z"}
  ]
}
```

For JSONL, each line is a message with `sender`, `text`, `timestamp`, and an optional
`conversation_id`. Optional external IDs and `reply_to` are preserved. Generic identity comes
from configured aliases, never an imported `is_user` assertion. Use explicit conversation IDs
when importing different datasets; absent IDs use a documented `default` conversation.

## What is measured

`persona analyze` writes `persona.md` and `persona-profile.json` in the private data directory and
stores structured profiles in SQLite. Only outgoing text feeds the analyzer. Incoming text is
kept as bounded retrieval context.

Measurements include character/word/sentence lengths, short message bursts, casing, punctuation,
emoji codepoints, repeated characters, common words and phrases, repeated slang/abbreviation
forms, greetings, sign-offs, Romanian diacritics, and Romanian/English word markers.
Language markers are heuristics, not a language classifier. Emoji counts measure codepoints,
not complete grapheme clusters. Repeated phrases/forms need at least three observations;
isolated misspellings are not instructions to add typos.

Context labels are explicit and extensible:

```sh
persona conversations
persona set-context CONVERSATION_ID business
persona set-context OTHER_ID casual
persona analyze
persona person "David"
```

Global profiles always use available outgoing text. Separate context/person profiles require at
least 20 messages by default. On-demand person statistics report their evidence count. No family,
dating, personality, sarcasm, or psychological classification is inferred. Recipient-specific
queries conservatively exclude groups. Exact names may occur on multiple platforms; pass a
platform to retrieval/writing-context when that distinction matters.

## Search and semantic retrieval

SQLite FTS5 searches outgoing messages and incoming interaction contexts. Interactions retain up
to three incoming messages and a burst of up to eight outgoing replies. A two-hour gap or a media
record breaks pairing; an outgoing burst spans at most five minutes between messages.

The local semantic provider uses `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
It embeds incoming contexts and stores float32 vectors in SQLite. Queries use exact cosine scans
over eligible vectors; this favors a simple local architecture over an external vector server.
Indexes resume after interruption. A recipient/context/platform filter applies before ranking.

If eligible vectors are absent, retrieval reports `lexical` explicitly. A partially built index
reports its coverage. Semantic scores below 0.25 are omitted; these scores are not probabilities
or guarantees of relevance. There is no silent cross-recipient fallback. Future providers can
implement the `EmbeddingProvider` protocol without changing the import or writing interfaces.

## MCP setup

The server uses the [official Python MCP SDK](https://github.com/modelcontextprotocol/python-sdk)
and stdio transport. Stdout carries only protocol messages; diagnostic output goes to stderr.

```sh
persona serve
```

Usually the client launches this command for you. Use **absolute paths** because the client's
working directory can differ from your terminal's. Example configuration for clients accepting
the common `mcpServers` structure:

```json
{
  "mcpServers": {
    "personamcp": {
      "command": "/absolute/path/to/venv/bin/persona",
      "args": ["--home", "/absolute/path/to/private/persona-data", "serve"]
    }
  }
}
```

On Windows the command is `C:\\absolute\\path\\.venv\\Scripts\\persona.exe`. For Codex:

```sh
codex mcp add personamcp -- /absolute/path/to/venv/bin/persona --home /absolute/path/to/private/persona-data serve
codex mcp list
```

See [Codex MCP setup](https://learn.chatgpt.com/docs/extend/mcp?surface=cli). Other clients may use
different configuration locations but need the same executable and arguments. Hosted clients
that only support remote HTTP MCP cannot directly launch this local stdio server. PersonaMCP
does not include a tunnel/HTTP bridge; exposing sensitive local data remotely requires a separate,
explicit deployment decision.

Tools:

| Tool | Result |
| --- | --- |
| `get_style_profile(context?)` | Measured global or assigned-context profile |
| `get_person_style(person)` | Communication statistics and evidence count for an exact person |
| `search_messages(query, person?, platform?, limit?)` | Bounded outgoing text matches |
| `find_similar_interactions(message, person?, context?, platform?, limit?)` | Similar real incoming/reply pairs |
| `get_writing_context(message, person?, context?, platform?)` | Appropriate style and up to three quoted examples |
| `get_persona_summary()` | Compact profile and database counts |

Historical content is returned inside `historical_quote`, with an explicit untrusted-data notice.
Tools supply evidence. Your agent generates the final reply and remains responsible for treating
historical instructions as data and for deciding what to send to its model provider.

## Skill setup

The reusable skill is [skills/write-like-me/SKILL.md](skills/write-like-me/SKILL.md). It is also
included in the wheel; `persona skill-path` prints its installed location. Copy its folder
to the skill directory your agent discovers. For current Codex repository discovery:

```sh
mkdir -p .agents/skills
cp -R skills/write-like-me .agents/skills/
```

Windows PowerShell: `New-Item -ItemType Directory -Force .agents/skills` followed by
`Copy-Item -Recurse skills/write-like-me .agents/skills/`. See
[Codex skill discovery](https://learn.chatgpt.com/docs/build-skills).

Then ask the connected agent to use `write-like-me`, for example:

> Write a short reply like me to Alex about “mai vii azi la cafea?”. Use PersonaMCP evidence.

The skill preserves supported casing, spelling, vocabulary, and length without forcing typos or
copying old facts. It can also use `persona writing-context` through a local shell, or an explicitly
supplied profile when MCP is unavailable. No global client configuration is changed by installation.

## Privacy and data controls

The default data directory comes from your operating system's application-data location.
`--home /private/path` or `PERSONAMCP_HOME` selects another location. Configuration is a local
`config.json`; the database uses SQLite foreign keys, versioned schema, and transactional imports.

```sh
persona stats --json
persona export-profile ./my-style.md
persona export-profile ./my-style.json --json
persona delete-person "Name"
persona delete-conversation CONVERSATION_ID
persona reset
```

Deletion/reset ask for confirmation; `--yes` is available for intentional scripting. Deleting a
person removes **entire conversations**, including groups containing that person, to avoid keeping
context about them. Profiles, FTS rows, and vectors are invalidated/purged and SQLite is vacuumed.
`reset` also clears configured owner identities but retains downloaded, non-personal model assets.
Original export files and any copied profiles remain in your control and are not deleted.

SQLite is not encrypted. Use a private data directory and disk encryption. Generated vocabulary
and examples are sensitive too. If your agent uses an external LLM, tool results may reach that
provider; “local-first” describes storage and computation, not the connected client's behavior.
Read [SECURITY.md](SECURITY.md) for deletion and prompt-injection limits.

## Offline benchmark

```sh
persona benchmark --limit 30
```

The last 20% of interactions by timestamp form a holdout. Retrieval and style profiles use only
earlier data. The held-out actual reply is used only for scoring. The default candidate is a
retrieved historical reply, **not** an LLM-generated answer.

The report includes retrieval coverage, response-length similarity, vocabulary overlap,
punctuation similarity, capitalization similarity, and their mean **Style Similarity Score**.
When a local model is available, embedding similarity is reported separately. No raw benchmark
replies are printed. `CandidateResponseProvider` is the extension point for a future explicitly
configured generator. Scores do not prove identity imitation, authorship, relevance, or generation
quality. Small or temporally uniform datasets cannot support this evaluation.

## Architecture

```text
exports → platform adapters → normalized conversations/messages
                                  ↓
                      SQLite + participants + FTS5
                                  ↓
                   bounded incoming/outgoing interactions
                         ↙                    ↘
              deterministic profiles      local embedding vectors
                         ↘                    ↙
                        retrieval/service layer
                            ↙           ↘
                         CLI          MCP stdio → writing skill → agent
```

The package uses a `src/personamcp` layout: importers, models, configuration, storage, analysis,
embeddings, retrieval, shared service, MCP server, CLI, and benchmark. No external database or
LLM provider is required. Schema compatibility is tracked with `PRAGMA user_version`; older
versions reject a newer database instead of guessing how to read it.

## Development, roadmap, and limits

Run `uv sync --locked`, then `uv run pytest`, `uv run ruff check src tests`,
`uv run ruff format --check src tests`, and `uv run mypy src`. CI checks Windows/Linux and Python
3.11–3.13, with no personal data or model download. Public fixtures are synthetic. See
[CONTRIBUTING.md](CONTRIBUTING.md), [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md), and
[implementation decisions](docs/IMPLEMENTATION.md).

Next steps are additional export adapters (Discord, Telegram, Messenger, iMessage, Signal),
explicit redaction rules, a faster local vector index for very large archives, richer language
signals, and optional generation-based evaluation. They are not implemented in this MVP.

This version is a CLI/MCP engine. Export schemas can change; group recipient inference,
psychological profiling, automatic typo correction, speech/media analysis, encryption at rest,
cloud embedding providers, HTTP hosting, and a frontend are outside its current support.

MIT licensed. Model weights and dependencies retain their own licenses; see the
[multilingual MiniLM model card](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
and [Sentence Transformers documentation](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html).
