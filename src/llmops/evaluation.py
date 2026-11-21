"""
Evaluation pipeline — an LLM's CI gate.

You cannot ship prompt/model changes on vibes. This runs a suite of cases through
any generation function and scores them with deterministic, explainable metrics
(exact-match, keyword-contains, must-not-contain, latency). It returns a report
with a pass rate you can gate a deployment on — the LLM analog of the repo's
champion-challenger promotion gate.

Deterministic scorers only (no LLM-as-judge here) so the eval itself is stable and
testable; an LLM-judge scorer plugs in as just another `Scorer`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

LLMFn = Callable[[str], str]


@dataclass
class EvalCase:
    name: str
    prompt: str
    expected_contains: List[str] = field(default_factory=list)   # all must appear
    must_not_contain: List[str] = field(default_factory=list)    # none may appear
    exact: Optional[str] = None


@dataclass
class CaseResult:
    name: str
    passed: bool
    latency_ms: float
    output: str
    failures: List[str] = field(default_factory=list)


def _score(case: EvalCase, output: str) -> List[str]:
    failures: List[str] = []
    low = output.lower()
    if case.exact is not None and output.strip() != case.exact.strip():
        failures.append("exact_mismatch")
    for kw in case.expected_contains:
        if kw.lower() not in low:
            failures.append(f"missing:{kw}")
    for kw in case.must_not_contain:
        if kw.lower() in low:
            failures.append(f"forbidden:{kw}")
    return failures


@dataclass
class EvalReport:
    total: int
    passed: int
    pass_rate: float
    p95_latency_ms: float
    results: List[CaseResult]

    def gate(self, min_pass_rate: float = 1.0) -> bool:
        """True if the suite meets the bar — call this in CI before promoting."""
        return self.pass_rate >= min_pass_rate

    def as_dict(self) -> dict:
        return {"total": self.total, "passed": self.passed,
                "pass_rate": round(self.pass_rate, 4),
                "p95_latency_ms": round(self.p95_latency_ms, 2),
                "results": [r.__dict__ for r in self.results]}


class EvaluationSuite:
    def __init__(self, cases: List[EvalCase]) -> None:
        self.cases = cases

    def run(self, llm: LLMFn) -> EvalReport:
        results: List[CaseResult] = []
        for case in self.cases:
            t0 = time.perf_counter()
            try:
                out = llm(case.prompt)
            except Exception as e:               # a crash is a failed case, not a crashed suite
                out = f"<error: {e}>"
            dt = (time.perf_counter() - t0) * 1000
            failures = _score(case, out)
            results.append(CaseResult(case.name, not failures, round(dt, 2), out, failures))

        passed = sum(r.passed for r in results)
        lat = sorted(r.latency_ms for r in results)
        p95 = lat[min(len(lat) - 1, int(0.95 * len(lat)))] if lat else 0.0
        return EvalReport(total=len(results), passed=passed,
                          pass_rate=(passed / len(results) if results else 0.0),
                          p95_latency_ms=p95, results=results)
