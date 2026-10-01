from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import PurePosixPath

from personamcp.models import Conversation, Message, timestamp


@dataclass
class Node:
    tag: str
    attrs: dict[str, str | None]
    children: list[Node | str] = field(default_factory=list)

    def text(self) -> str:
        return "".join(c.text() if isinstance(c, Node) else c for c in self.children)

    def find(self, tag: str | None = None, css: str | None = None) -> list[Node]:
        nodes = []
        for child in self.children:
            if isinstance(child, Node):
                if (tag is None or child.tag == tag) and (
                    css is None or css in (child.attrs.get("class") or "").split()
                ):
                    nodes.append(child)
                nodes.extend(child.find(tag, css))
        return nodes


class Tree(HTMLParser):
    """Passive HTML tree. No JavaScript, network, media, or browser execution."""

    def __init__(self, source: str):
        super().__init__(convert_charrefs=True)
        self.root = Node("root", {})
        self.stack = [self.root]
        self.feed(source)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        node = Node(tag, dict(attrs))
        self.stack[-1].children.append(node)
        if tag not in {"img", "br", "hr", "meta", "link", "input", "path"}:
            if len(self.stack) > 256:
                raise ValueError("HTML nesting exceeds safe depth")
            self.stack.append(node)
        elif tag == "br":
            node.children.append("\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.stack[-1].children.append(Node(tag, dict(attrs)))

    def handle_endtag(self, tag: str) -> None:
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i].tag == tag:
                del self.stack[i:]
                return

    def handle_data(self, data: str) -> None:
        self.stack[-1].children.append(data)


def instagram_html(text: str, source: str) -> list[Conversation]:
    root = Tree(text).root
    messages: list[Message] = []
    page_match = re.search(r"message_(\d+)\.html$", source)
    page = int(page_match.group(1)) if page_match else 1
    for block in root.find(css="_a6-g"):
        sender = block.find(css="_a6-i")
        dates = block.find(css="_a6-o")
        bodies = block.find(css="_a6-p")
        if not sender or not dates:
            continue
        body = ""
        if bodies:
            # Meta places attachments next to the first nested text div. Do not index filenames.
            divs = [n for n in bodies[0].children if isinstance(n, Node) and n.tag == "div"]
            if divs:
                inner = [n for n in divs[0].children if isinstance(n, Node) and n.tag == "div"]
                if inner:
                    # Current Meta exports: empty metadata, authored text, attachments, reactions.
                    authored = inner[1] if len(inner) >= 4 else inner[0]
                    body = authored.text().strip()
                    if re.search(r"(?:sent an attachment|shared a story)\.?$", body):
                        body = ""
        messages.append(
            Message(
                sender[0].text().strip(),
                body,
                timestamp(dates[0].text().strip()),
                metadata={
                    "kind": "text" if body else "nontext",
                    "export_sequence": -(page * 1_000_000_000 + len(messages)),
                },
            )
        )
    if not messages:
        raise ValueError("No supported Instagram HTML messages found")
    external = str(PurePosixPath(source).parent)
    title_nodes = root.find(css="_a70e")
    title = title_nodes[0].text().strip() if title_nodes else external
    return [
        Conversation("instagram", external, title, sorted({m.sender for m in messages}), messages)
    ]


def snapchat_html(text: str, source: str) -> list[Conversation]:
    root = Tree(text).root
    panels = root.find(css="rightpanel")
    if not panels:
        raise ValueError("Unsupported Snapchat HTML layout")
    if "subpage_" not in PurePosixPath(source).name:
        return []  # Index pages point at subpages; no browser navigation needed.
    messages = []
    for node in panels[0].find(tag="div"):
        if "background: #f2f2f2" not in (node.attrs.get("style") or ""):
            continue
        senders, dates, bodies = node.find("h4"), node.find("h6"), node.find("p")
        if not senders or not dates:
            raise ValueError("Unsupported Snapchat chat card")
        date = dates[0].text().strip()
        date = re.sub(r"^(Created|Date):\s*", "", date)
        body = bodies[0].text().strip() if bodies else ""
        messages.append(
            Message(
                senders[0].text().strip(),
                body,
                timestamp(date),
                metadata={"kind": "text" if body else "nontext"},
            )
        )
    if not messages:
        return []
    person = PurePosixPath(source).stem.removeprefix("subpage_")
    return [
        Conversation(
            "snapchat", person, person, sorted({person, *(m.sender for m in messages)}), messages
        )
    ]
