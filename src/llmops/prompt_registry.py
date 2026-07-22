"""
Prompt Registry — versioned, aliased prompt templates.

Prompts are production artifacts: a wording change can regress quality as badly as
a model swap, so they deserve the same discipline as models — immutable versions,
aliases (`champion`/`canary`), and a content hash for auditability. This mirrors
`src/platform/registry.py` deliberately; a prompt is just a model of a different
shape.
"""
from __future__ import annotations

import hashlib
import string
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


class _SafeDict(dict):
    """str.format_map helper: leave unknown {placeholders} untouched rather than
    raising, so a template can be rendered with a partial variable set."""
    def __missing__(self, key):
        return "{" + key + "}"


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    version: int
    template: str
    description: str = ""
    created_at: float = field(default_factory=time.time)
    aliases: frozenset = field(default_factory=frozenset)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.template.encode()).hexdigest()[:16]

    @property
    def variables(self) -> List[str]:
        """The {placeholders} this template expects."""
        return sorted({fn for _, fn, _, _ in string.Formatter().parse(self.template) if fn})

    def render(self, **kwargs) -> str:
        return self.template.format_map(_SafeDict(**kwargs))

    def to_public(self) -> dict:
        return {"name": self.name, "version": self.version, "description": self.description,
                "sha256": self.sha256, "variables": self.variables,
                "aliases": sorted(self.aliases), "created_at": self.created_at}


class PromptRegistry:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._versions: Dict[str, Dict[int, PromptTemplate]] = {}
        self._aliases: Dict[tuple, int] = {}

    def register(self, name: str, template: str, description: str = "") -> PromptTemplate:
        with self._lock:
            vers = self._versions.setdefault(name, {})
            version = (max(vers) + 1) if vers else 1
            pt = PromptTemplate(name=name, version=version, template=template,
                                description=description)
            vers[version] = pt
            return pt

    def get(self, name: str, selector: str = "latest") -> Optional[PromptTemplate]:
        with self._lock:
            vers = self._versions.get(name)
            if not vers:
                return None
            if selector == "latest":
                return vers[max(vers)]
            if str(selector).lstrip("v").isdigit():
                return vers.get(int(str(selector).lstrip("v")))
            v = self._aliases.get((name, selector))
            return vers.get(v) if v is not None else None

    def set_alias(self, name: str, version: int, alias: str) -> PromptTemplate:
        with self._lock:
            if version not in self._versions.get(name, {}):
                raise KeyError(f"{name} v{version} not registered")
            self._aliases[(name, alias)] = version
            self._reindex(name)
            return self._versions[name][version]

    def render(self, name: str, selector: str = "champion", /, **kwargs) -> str:
        # name/selector are positional-only (the `/`) so a template variable also
        # named "name" or "selector" can be passed as a kwarg without colliding.
        pt = self.get(name, selector)
        if pt is None:
            raise KeyError(f"prompt '{name}' selector '{selector}' not found")
        return pt.render(**kwargs)

    def list(self) -> List[PromptTemplate]:
        with self._lock:
            return [pt for vers in self._versions.values() for pt in vers.values()]

    def _reindex(self, name: str) -> None:
        per_version: Dict[int, set] = {}
        for (n, alias), v in self._aliases.items():
            if n == name:
                per_version.setdefault(v, set()).add(alias)
        for v, pt in list(self._versions.get(name, {}).items()):
            new_aliases = frozenset(per_version.get(v, set()))
            if new_aliases != pt.aliases:
                self._versions[name][v] = PromptTemplate(
                    name=pt.name, version=pt.version, template=pt.template,
                    description=pt.description, created_at=pt.created_at, aliases=new_aliases)
