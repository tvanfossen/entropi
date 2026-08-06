#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Requirements-traceability gate for docs/requirements.yaml.

## @brief Verify every @req tag resolves and every requirement is implemented.
#  @version 2.10.4
#  @req REQ-API-014
#  @utility

Three checks, run as one pre-commit gate. Each closes a decay path that
actually happened in this repo:

1. ORPHAN REFS — every ``@req REQ-*`` in source must exist in the catalog.
   doxygen-guard has this check, but it SKIPS BODILESS HEADER DECLARATIONS,
   so ``i_inference_backend.h`` carried a dead ``REQ-INFER-003`` that no gate
   could see. This scan is textual and therefore catches declarations too.

2. UNCOVERED — every catalog entry must be referenced by at least one
   ``@req`` in source. A requirement nothing implements is a claim without
   code; a catalog full of them is how the previous 140-entry catalog became
   decorative (10 of 140 referenced) before being deleted outright in
   ``2edfb4e``.

3. EXEMPTION RATIO — ``@req`` must remain a meaningful share of all tagged
   functions. This is the check nothing else can make: doxygen-guard's
   coverage command never collects a function whose only tag is
   ``@internal``/``@utility``/``@callback``, so a wholesale drift back toward
   exemptions is invisible to it. The repo sat at 21 ``@req`` against 2318
   exemptions (0.9%) with every gate green.

A ratio floor rather than an absolute count: deleting code legitimately
lowers both numbers, and only the proportion expresses the property worth
defending.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

CATALOG = Path("docs/requirements.yaml")
ROOTS = ("src", "include", "scripts", "python/src")
SUFFIXES = (".h", ".hpp", ".cpp", ".cc", ".cxx", ".py")
SKIP = ("_bindings.py", "_bindings_manifest.py")

REQ_TAG = re.compile(r"@req\s+(REQ-[A-Z]+-\d+)")
EXEMPT_TAG = re.compile(r"@(internal|utility|callback)\b")

#: Minimum share of tagged functions that must carry a real @req.
#: Measured 51.8% when the catalog was completed; 50 leaves headroom for
#: ordinary churn while still failing a slide back toward blanket exemption.
MIN_REQ_RATIO = 0.50


## @brief Yield every source file the gate inspects.
#  @version 2.10.4
#  @return Iterator of Path objects.
#  @utility
def source_files():
    for root in ROOTS:
        for path in Path(root).rglob("*"):
            if path.suffix not in SUFFIXES:
                continue
            if path.name in SKIP:
                continue
            yield path


## @brief Collect requirement-tag ids and tag counts from the tree.
#  @version 2.10.4
#  @return Tuple of (ids -> sorted files citing them, req count, exempt count).
#  @utility
def scan_source() -> tuple[dict[str, list[str]], int, int]:
    cited: dict[str, set[str]] = {}
    n_req = 0
    n_exempt = 0
    for path in source_files():
        text = path.read_text(errors="replace")
        for match in REQ_TAG.finditer(text):
            cited.setdefault(match.group(1), set()).add(str(path))
            n_req += 1
        n_exempt += len(EXEMPT_TAG.findall(text))
    return {k: sorted(v) for k, v in cited.items()}, n_req, n_exempt


## @brief Report requirement-tag ids that name no catalog entry.
#  @version 2.10.4
#  @return Number of violations found.
#  @utility
def check_orphans(cited: dict[str, list[str]], known: set[str]) -> int:
    orphans = {rid: files for rid, files in cited.items() if rid not in known}
    if not orphans:
        return 0
    print(f"\nORPHAN @req ids ({len(orphans)}) — not in {CATALOG}:")
    for rid, files in sorted(orphans.items()):
        print(f"  {rid}")
        for f in files:
            print(f"      {f}")
    return len(orphans)


## @brief Report catalog entries no source file references.
#  @version 2.10.4
#  @return Number of violations found.
#  @utility
def check_uncovered(cited: dict[str, list[str]], known: set[str]) -> int:
    uncovered = sorted(known - set(cited))
    if not uncovered:
        return 0
    print(f"\nUNCOVERED requirements ({len(uncovered)}) — no @req in source:")
    for rid in uncovered:
        print(f"  {rid}")
    return len(uncovered)


## @brief Report a slide back toward blanket exemption tags.
#  @version 2.10.4
#  @return 1 when the ratio floor is breached, else 0.
#  @utility
def check_ratio(n_req: int, n_exempt: int) -> int:
    total = n_req + n_exempt
    ratio = (n_req / total) if total else 0.0
    print(
        f"\n@req {n_req} / exemptions {n_exempt} — ratio {ratio:.1%} "
        f"(floor {MIN_REQ_RATIO:.0%})"
    )
    if ratio >= MIN_REQ_RATIO:
        return 0
    print(
        "  Exemption creep: functions are being marked @internal/@utility/"
        "@callback instead of linked to a requirement."
    )
    return 1


## @brief Run all three traceability checks.
#  @version 2.10.4
#  @return Process exit code: 0 when every check passes, 1 otherwise.
#  @utility
def main() -> int:
    if not CATALOG.exists():
        print(f"FAIL: {CATALOG} is missing.")
        print("  It was deleted once before (2edfb4e) and nothing noticed for")
        print("  15 months because every consumer fails open when it is absent.")
        return 1

    known = {e["id"] for e in yaml.safe_load(CATALOG.read_text())}
    cited, n_req, n_exempt = scan_source()

    print(f"Requirements traceability against {CATALOG}")
    print(f"  catalog entries : {len(known)}")
    print(f"  ids cited       : {len(cited)}")

    failures = check_orphans(cited, known)
    failures += check_uncovered(cited, known)
    failures += check_ratio(n_req, n_exempt)

    if failures:
        print(f"\nFAILED — {failures} traceability violation(s).")
        return 1
    print("\nPASS — every @req resolves, every requirement is implemented.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
