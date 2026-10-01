from __future__ import annotations

from pathlib import Path

from personamcp.config import Config
from personamcp.importers import InstagramImporter
from personamcp.storage import Store


def card(sender: str, body: str) -> str:
    return (
        f"<div class='_a6-g'><div class='_a6-i'>{sender}</div><div class='_a6-p'>"
        f"<div><div></div><div>{body}</div><div></div><div></div></div></div>"
        "<div class='_a6-o'>Jun 26, 2024, 6:59 AM</div></div>"
    )


def test_meta_reverse_order_at_same_minute_across_pages(tmp_path: Path) -> None:
    adapter = InstagramImporter()
    # Meta lists newest first. A minute boundary can be split across two export pages.
    newer = adapter.parse(
        card("Owner", "da vin acu") + card("Owner", "poate ies"), "inbox/alex/message_1.html"
    )
    older = adapter.parse(card("Alex", "ce faci azi?"), "inbox/alex/message_2.html")
    store = Store(tmp_path)
    store.import_conversations(newer + older, "pages", "instagram", Config(name="Owner"))
    rows = store.rows("SELECT * FROM interactions")
    assert len(rows) == 1
    assert rows[0]["incoming"] == "ce faci azi?"
    assert rows[0]["user_reply"] == "poate ies\nda vin acu"
    store.close()
