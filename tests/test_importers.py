from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from personamcp.importers import (
    GenericImporter,
    InstagramImporter,
    SnapchatImporter,
    WhatsAppImporter,
    importer_for,
    meta_text,
    read_sources,
)
from personamcp.models import timestamp


def test_instagram_json_unicode_media_and_pagination() -> None:
    data = {
        "participants": [{"name": "Owner"}, {"name": "Alex"}],
        "thread_path": "messages/inbox/alex_123",
        "title": "Alex",
        "messages": [
            {
                "sender_name": "Owner",
                "timestamp_ms": 1704067200000,
                "content": "mersi 😄".encode().decode("latin1"),
            },
            {
                "sender_name": "Alex",
                "timestamp_ms": 1704067199000,
                "photos": [{"uri": "never-load.jpg"}],
            },
        ],
    }
    first = InstagramImporter().parse(json.dumps(data), "inbox/alex_123/message_1.json")[0]
    second = InstagramImporter().parse(json.dumps(data), "inbox/alex_123/message_2.json")[0]
    assert first.id == second.id
    assert first.messages[0].text == "mersi 😄"
    assert first.messages[1].metadata["kind"] == "nontext"
    assert meta_text("bună") == "bună"


def test_instagram_html_does_not_execute_scripts() -> None:
    source = """<div class='_a70e'>Alex</div><script>fetch('https://evil.test')</script>
    <div class='_a6-g'><div class='_a6-i'>Owner</div><div class='_a6-p'>
    <div><div>acu &amp; gen<br>vin</div><div><img src='https://evil.test/private'></div></div>
    </div><div class='_a6-o'>Jun 26, 2024, 6:59 AM</div></div>"""
    conv = InstagramImporter().parse(source, "inbox/alex_123/message_1.html")[0]
    assert conv.messages[0].text == "acu & gen\nvin"
    assert conv.messages[0].timestamp.startswith("2024-06-26T06:59")


def test_instagram_current_html_text_slot_excludes_attachment_metadata() -> None:
    source = """<div class='_a6-g'><div class='_a6-i'>Owner</div><div class='_a6-p'>
    <div><div></div><div>gen acu vin</div><div>secret-media-filename.jpg</div><div></div></div>
    </div><div class='_a6-o'>Jun 26, 2024, 6:59 AM</div></div>"""
    conv = InstagramImporter().parse(source, "inbox/alex/message_1.html")[0]
    assert conv.messages[0].text == "gen acu vin"


@pytest.mark.parametrize("nested", [True, False])
def test_snapchat_json_layouts(nested: bool) -> None:
    chats = {
        "Alex": [
            {
                "From": "Owner",
                "Created": "2024-01-01 10:00:00 UTC",
                "Content": "gen acu",
                "Media Type": "TEXT",
            }
        ]
    }
    conv = SnapchatImporter().parse(
        json.dumps({"Chat History": chats} if nested else chats), "chat_history.json"
    )[0]
    assert conv.messages[0].text == "gen acu"
    assert conv.platform == "snapchat"


def test_snapchat_html_cards() -> None:
    html = """<div class='rightpanel'><h1>Alex</h1>
    <div style='background: #f2f2f2;'><span><div><h4>Owner</h4></div>
    <p>da =))</p><h6>2024-01-01 10:00:00 UTC</h6></span></div></div>"""
    conv = SnapchatImporter().parse(html, "html/chat_history/subpage_Alex.html")[0]
    assert conv.external_id == "Alex"
    assert conv.messages[0].sender == "Owner"
    assert conv.messages[0].text == "da =))"


@pytest.mark.parametrize(
    "line",
    [
        "01/02/2024, 10:30 - Owner: prima linie",
        "[01/02/2024, 10:30:00] Owner: prima linie",
        "[01/02/2024, 10:30:00 AM] Owner: prima linie",
    ],
)
def test_whatsapp_multiline_and_notices(line: str) -> None:
    conv = WhatsAppImporter().parse(
        "01/02/2024, 10:00 - Messages are encrypted\n" + line + "\na doua linie", "chat.txt"
    )[0]
    assert len(conv.messages) == 1
    assert conv.messages[0].text == "prima linie\na doua linie"
    assert conv.messages[0].timestamp.startswith("2024-02-01")


def test_whatsapp_month_first() -> None:
    conv = WhatsAppImporter(day_first=False).parse("02/15/24, 10:30 - Owner: ok", "chat.txt")[0]
    assert conv.messages[0].timestamp.startswith("2024-02-15")


@pytest.mark.parametrize("suffix", ["json", "jsonl"])
def test_generic_format(suffix: str) -> None:
    rows = [
        {
            "sender": "Owner",
            "text": "acu",
            "timestamp": "2024-01-01T00:00:00Z",
            "conversation_id": "a",
        },
        {"sender": "Alex", "text": "hello", "timestamp": 1704067201000, "conversation_id": "b"},
    ]
    text = json.dumps(rows) if suffix == "json" else "\n".join(json.dumps(r) for r in rows)
    assert len(GenericImporter().parse(text, f"messages.{suffix}")) == 2


@pytest.mark.parametrize(
    "text", ['{"messages":', "null", '["not an object"]', '{"messages":[{"sender":"Owner"}]}']
)
def test_malformed_generic_has_clear_error(text: str) -> None:
    with pytest.raises(ValueError):
        GenericImporter().parse(text, "test.json")


def test_zip_prefers_json_and_does_not_extract(tmp_path: Path) -> None:
    path = tmp_path / "chat.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr("json/chat_history.json", "{}")
        archive.writestr("html/chat_history/subpage_Alex.html", "duplicate")
        archive.writestr("media/secret.jpg", "never-read")
    assert list(read_sources(path, "snapchat")) == [("json/chat_history.json", "{}")]
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize(
    "name", ["../chat_history.json", "/chat_history.json", "C:/chat_history.json"]
)
def test_unsafe_zip_paths_rejected(tmp_path: Path, name: str) -> None:
    path = tmp_path / "chat.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr(name, "{}")
    with pytest.raises(ValueError, match="Unsafe"):
        list(read_sources(path, "snapchat"))


def test_oversize_source_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "chat.json"
    path.write_text("{}")
    monkeypatch.setattr("personamcp.importers.MAX_FILE_BYTES", 1)
    with pytest.raises(ValueError, match="size"):
        list(read_sources(path, "json"))


def test_unsupported_platform() -> None:
    with pytest.raises(ValueError, match="Supported"):
        importer_for("discord")


@pytest.mark.parametrize(
    "platform,data",
    [
        ("instagram", {"participants": ["not an object"], "messages": []}),
        ("instagram", {"messages": [{"sender_name": 42, "content": "x"}]}),
        ("snapchat", {"Alex": [None]}),
        ("snapchat", {"Alex": [{"From": {"bad": True}, "Created": "2024-01-01"}]}),
        ("json", {"conversations": {"bad": True}}),
    ],
)
def test_wrong_structural_types_rejected(platform: str, data: object) -> None:
    with pytest.raises(ValueError):
        importer_for(platform).parse(json.dumps(data), "chat.json")


@pytest.mark.parametrize(
    "value", [1704067200, 1704067200000, "2024-01-01T02:00:00+02:00", "2024-01-01 00:00:00 UTC"]
)
def test_timestamp_normalization(value: object) -> None:
    assert timestamp(value) == "2024-01-01T00:00:00+00:00"
