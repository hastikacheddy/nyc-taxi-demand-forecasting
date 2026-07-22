"""
Guardrails — input and output safety for LLM traffic.

An LLM endpoint is an untrusted-input, untrusted-output surface (OWASP LLM Top 10,
see docs/platform/SECURITY.md §2.2). Guardrails are the enforceable checks around
it:

  Input:  PII leakage, prompt-injection heuristics, oversized prompts.
  Output: PII echo, blocklisted content, oversized/empty responses.

Each check returns a structured verdict rather than raising, so the caller can
decide policy (block / redact / warn). PII detection *reuses* the repo's existing
scanner patterns (src/data/pii.py) — one definition of PII across the whole system.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from src.data.pii import PATTERNS as PII_PATTERNS


@dataclass
class GuardrailResult:
    allowed: bool
    stage: str                       # "input" | "output"
    violations: List[str] = field(default_factory=list)
    redacted_text: Optional[str] = None

    def as_dict(self) -> dict:
        return {"allowed": self.allowed, "stage": self.stage,
                "violations": self.violations, "redacted": self.redacted_text is not None}


# ── Individual checks ──────────────────────────────────────────────
def find_pii(text: str) -> List[str]:
    return [name for name, pat in PII_PATTERNS.items() if pat.search(text)]


def redact_pii(text: str) -> str:
    out = text
    for name, pat in PII_PATTERNS.items():
        out = pat.sub(f"[REDACTED_{name.upper()}]", out)
    return out


# Heuristic prompt-injection signatures. Not a complete defence (nothing is) — a
# fast first line that catches the common "ignore previous instructions" family.
_INJECTION_SIGNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions", re.I),
    re.compile(r"disregard\s+(the\s+)?(system|previous)\s+prompt", re.I),
    re.compile(r"you\s+are\s+now\s+(a\s+)?(dan|developer mode|unrestricted)", re.I),
    re.compile(r"reveal\s+(your\s+)?(system\s+prompt|instructions)", re.I),
    re.compile(r"\bprint\s+your\s+(system\s+)?prompt\b", re.I),
]


def detect_injection(text: str) -> bool:
    return any(p.search(text) for p in _INJECTION_SIGNS)


# ── Pipeline ───────────────────────────────────────────────────────
@dataclass
class GuardrailPipeline:
    max_input_chars: int = 8000
    max_output_chars: int = 16000
    block_injection: bool = True
    redact_input_pii: bool = True
    block_output_pii: bool = True
    output_blocklist: List[str] = field(default_factory=list)
    # extra custom checks: text -> violation name or None
    custom_input_checks: List[Callable[[str], Optional[str]]] = field(default_factory=list)

    def check_input(self, text: str) -> GuardrailResult:
        violations: List[str] = []
        redacted: Optional[str] = None

        if len(text) > self.max_input_chars:
            violations.append(f"input_too_long>{self.max_input_chars}")

        if self.block_injection and detect_injection(text):
            violations.append("prompt_injection")

        pii = find_pii(text)
        if pii:
            if self.redact_input_pii:
                redacted = redact_pii(text)
                violations.append("pii_redacted:" + ",".join(pii))
            else:
                violations.append("pii:" + ",".join(pii))

        for chk in self.custom_input_checks:
            v = chk(text)
            if v:
                violations.append(v)

        # Blocking violations: injection or over-length. PII-redaction is allowed
        # (we sanitized it), so it does not block on its own.
        blocking = any(v.startswith(("prompt_injection", "input_too_long")) or
                       v.startswith("pii:") for v in violations)
        return GuardrailResult(allowed=not blocking, stage="input",
                               violations=violations, redacted_text=redacted)

    def check_output(self, text: str) -> GuardrailResult:
        violations: List[str] = []
        if not text.strip():
            violations.append("empty_output")
        if len(text) > self.max_output_chars:
            violations.append(f"output_too_long>{self.max_output_chars}")
        if self.block_output_pii and find_pii(text):
            violations.append("output_pii")
        low = text.lower()
        for term in self.output_blocklist:
            if term.lower() in low:
                violations.append(f"blocked_term:{term}")
        return GuardrailResult(allowed=len(violations) == 0, stage="output",
                               violations=violations)
