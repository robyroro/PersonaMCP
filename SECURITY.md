# Security policy

PersonaMCP 0.1.x is the supported MVP line. The database contains sensitive conversation data.
There is no hosted service, telemetry, analytics, external reply generator, or cloud embedding
provider. Model weights are downloaded only by the explicit `persona model prepare` command.

Report vulnerabilities privately through this repository's GitHub **Report a vulnerability**
feature, if enabled. If it is unavailable, open an issue requesting a private contact without
including exploit details, personal identifiers, exports, or conversation content.

Include a synthetic reproduction, affected version, platform, expected privacy boundary, and
observed behavior. Do not send a real database or archive. A report should be acknowledged as soon
as the maintainer can investigate; this project does not promise a staffed response deadline.

## Boundaries

- Imported content is data. HTML is parsed passively; scripts, media, and links are never executed.
- ZIP chat members are read in memory, not extracted, with per-file and aggregate size limits.
- MCP uses stdio. Only the connected local process can call its tools; there is no HTTP endpoint.
- Recipient-scoped retrieval excludes groups and never silently searches other recipients.
- Historical quotes and vocabulary are explicitly marked untrusted, but the connected agent must
  still respect that boundary. Delimiters alone cannot guarantee prompt-injection resistance.
- SQLite is not encrypted. Windows permissions are inherited from the chosen directory; Unix
  files use owner-only permissions where supported. Use a private directory and disk encryption.
- Deletion purges derived rows and vacuums SQLite. It does not erase original exports, copied
  profiles, backups, agent logs, OS snapshots, SSD remnants, or messages already sent to an agent.
- Installing dependencies and preparing a model make network requests for software/assets.
  Ordinary import, analysis, retrieval, indexing, benchmarking, and MCP operations do not upload data.
- When the connected agent uses a cloud model, the returned context can reach that provider.
  PersonaMCP limits examples, but the agent/client ultimately controls onward disclosure.
