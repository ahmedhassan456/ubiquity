"""A survey of whether every configuration field is read by anything.

A configuration field that no code reads is a promise the SDK does not keep,
and it fails silently: the caller sets it, sees no error, and gets a run
configured as though they had not. This is the check that catches that, shared
by the offline suite and the live one so the two cannot disagree.

A field counts as read when some code outside the class that declares it
mentions the name, or when a method of that class reads it and something calls
that method.

The audit is not specific to `Options`. Any dataclass a caller fills in makes
the same promise, and the fields that went unread longest were the ones in the
classes this check did not look at.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

SDK_ROOT = Path(__file__).resolve().parents[1] / "src" / "ubiquity"


def _class_node(source: str, name: str) -> ast.ClassDef:
    """Return the `ast` node for the class named `name`."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise LookupError(f"{name} is not defined in the given source")


def _methods_reading(node: ast.ClassDef, source: str) -> dict[str, set[str]]:
    """Map each field to the methods of `node` that read it.

    Only this class's own methods are scanned. Attributing a sibling class's
    `self.x` to it would let one class's field be kept alive by another's,
    which is the same silent promise the audit exists to catch.
    """
    reading: dict[str, set[str]] = {}
    for member in node.body:
        if not isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for inner in ast.walk(member):
            if (
                isinstance(inner, ast.Attribute)
                and isinstance(inner.value, ast.Name)
                and inner.value.id == "self"
            ):
                reading.setdefault(inner.attr, set()).add(member.name)
    return reading


def _strip_self(source: str) -> str:
    """Remove `self.x` accesses, which never vouch for another class's field.

    Outside code reads a field through a variable -- `ctx.abort`,
    `options.model`. A `self.x` belongs to whatever class encloses it, and
    counting it here is how a field of one class gets kept alive by an
    unrelated field of the same name on another. A class reading its own field
    is covered precisely by `_methods_reading` instead.
    """
    return re.sub(r"\bself\.\w+", "", source)


def unread_fields(cls: type, module: Path, root: Path = SDK_ROOT) -> list[str]:
    """Return the dataclass fields of `cls` that nothing in the SDK reads.

    `module` is the file that declares `cls`. Its text is searched like any
    other, minus the class's own body: a field is not kept alive by the class
    that declares it, but the rest of that module is ordinary calling code.
    """
    source = module.read_text(encoding="utf-8")
    node = _class_node(source, cls.__name__)
    reading = _methods_reading(node, source)

    lines = source.splitlines(keepends=True)
    start = node.lineno - 1
    end = node.end_lineno if node.end_lineno is not None else len(lines)
    outside = "".join(lines[:start] + lines[end:])

    sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in root.rglob("*.py")
        if path.resolve() != module.resolve()
    )
    sources = _strip_self(sources + "\n" + outside)

    unread: list[str] = []
    for field in cls.__dataclass_fields__:
        if f".{field}" in sources or f'"{field}"' in sources:
            continue
        if any(f"{method}(" in sources for method in reading.get(field, ())):
            continue
        unread.append(field)
    return sorted(unread)


def unread_option_fields(root: Path = SDK_ROOT) -> list[str]:
    """Return the `Options` fields nothing in the SDK reads."""
    import sys

    sys.path.insert(0, str(root.parent))
    from ubiquity import Options

    return unread_fields(Options, root / "options.py", root)


def audited_classes(root: Path = SDK_ROOT) -> list[tuple[type, Path]]:
    """Return every caller-facing dataclass the audit covers, with its module.

    Listed explicitly rather than discovered. A class that a caller fills in
    makes the promise this audit checks; one the SDK fills in for itself does
    not, and discovery cannot tell the two apart.
    """
    import sys

    sys.path.insert(0, str(root.parent))
    from ubiquity import AgentDefinition, Options
    from ubiquity.tool import ToolContext

    return [
        (Options, root / "options.py"),
        (AgentDefinition, root / "options.py"),
        (ToolContext, root / "tool.py"),
    ]


__all__ = ["unread_fields", "unread_option_fields", "audited_classes", "SDK_ROOT"]
