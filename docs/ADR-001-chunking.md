# ADR-001: Section-Aware Chunking

## Status
Accepted

## Context
FR-2 requires the chunking strategy to be "a deliberate, documented decision
justified against your document structure," not a default. The target
corpus (specs, policy, curricula) is structured around headings and
numbered clauses — `FR-2`, `4. Architecture`, `## Module 3`. A naive fixed
character-window split will frequently sever a requirement's body from its
own clause number, which directly damages two things the spec measures:
citation traceability (FR-2) and groundedness (FR-3).

## Decision
Two-phase chunking:
1. **Section split** — a heading regex detects markdown headers, `FR-/ADR-/NFR-###`
   style clause labels, numbered sections (`4.`, `4.2.`), and short
   Title-Case lines that look like headings. Text is split at these
   boundaries first, so no chunk ever straddles two unrelated requirements.
2. **Sliding window within a section** — if a section is still longer than
   `max_chars` (default 800, overlap 150), it's split on sentence
   boundaries with overlap, so a chunk never starts or ends mid-sentence
   and adjacent chunks share enough context that a citation near a window
   edge isn't stranded.

Every chunk carries its section label, ordinal position, and exact
character offsets into the source document — the traceability FR-2 asks
for.

## Consequences
- **Positive**: citations map cleanly to the clause they came from;
  retrieval doesn't need to guess where a requirement "really" starts.
- **Negative / limitation**: the heading regex is heuristic. A document
  with no discernible structure (e.g. a wall of prose with no headings)
  falls back to one giant "section" that then gets sliding-windowed —
  functionally equivalent to naive chunking in that case. This is
  acceptable because the target corpus is structured; it's called out here
  rather than hidden.
- Chunk sizes are therefore uneven by design (a whole short clause vs. a
  windowed slice of a long one) — downstream retrieval scoring must not
  assume uniform chunk length.

## Alternatives considered
- **Fixed-size character/token windows** — simplest, but rejected: it
  regularly cuts a requirement ID away from its own text, which is exactly
  the failure mode the spec's evaluation set (FR-3) is designed to catch.
- **One chunk per paragraph** — rejected: many clauses in this corpus are
  themselves multi-paragraph, so this either over-fragments short clauses
  or under-splits long ones without a size bound.
