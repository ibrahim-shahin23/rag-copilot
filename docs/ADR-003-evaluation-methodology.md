# ADR-003: Evaluation Harness Methodology

## Status
Accepted

## Context
FR-3 requires a golden set of ≥25 Q/A pairs including ≥5 adversarial cases
(out-of-corpus, ambiguous, prompt injection, conflicting sources), and a
runnable harness reporting retrieval hit-rate, groundedness, and refusal
correctness — with actual baseline numbers recorded, bad ones included.

## Decision: fully offline, deterministic harness
`eval/harness.py` runs against the same offline adapter stack the rest of
the test suite uses (TF-IDF embeddings, BM25, SQLite, extractive-fallback
LLM) — not a hosted API. This means the harness needs no API key and
produces bit-for-bit reproducible numbers, so it can run on every PR in
CI, not just occasionally on a developer's machine with a key configured.
The cost is that groundedness numbers reflect the offline embedder's
retrieval quality, not a production embedding model's — that's an
explicit, documented ceiling on what this run can tell you (see
`eval/results/report.md`'s Interpretation section), not something the
harness hides.

## Decision: separate "retrieval hit" from "refusal correctness"
`AnswerQueryUseCase.retrieve()` was split out from `.execute()`
specifically to make this possible (see `application/retrieve.py`).
Conflating them would make "the system refused" and "the system never
retrieved anything relevant" indistinguishable — two very different
failure modes that need different fixes (a threshold problem vs. a
chunking/embedding problem).

## Decision: groundedness measured against retrieved chunk text, not answer text
Groundedness for a golden item is the fraction of `expected_answer_contains`
keywords found in the *cited chunks'* text, not in the LLM's generated
answer text. This deliberately measures "was the evidence actually
present in what got retrieved," independent of how faithfully a given LLM
paraphrases it — a real LLM's paraphrasing quality is a separate concern
from retrieval quality, and this harness is testing FR-2's retrieval
pipeline, not a specific LLM's summarization. (The extractive-fallback
provider used here happens to just echo the top excerpt verbatim, which
makes this distinction somewhat moot for *this* run — but it stays
correct once a real LLM is wired in and starts paraphrasing.)

## Decision: multi-source expected_sources means "all of them," not "any of them"
For the conflicting-sources and ambiguous-term items, `expected_sources`
lists both documents, and the hit condition requires both to appear in
the retrieved set. A single-source hit would silently accept "the system
picked one side of a disagreement and never showed the other" as a pass,
which is the opposite of what a conflicting-sources test should reward.

## Decision: prompt-injection items score leak-freedom, not refusal
`expected_refusal: null` for the two injection items — there's no
"correct" refuse-or-answer call for an injection attempt in the abstract;
the actual requirement is that the injected instruction never causes the
real system prompt to leak into the answer (`must_not_contain`). Scoring
these on refusal-correctness instead would conflate "is this system
prompt-injection-resistant" with "is this question in-corpus," which are
different questions.

## Consequences / limitations, stated plainly
- The corpus is small (6 documents, 17 chunks) and synthetic — built
  specifically to exercise each adversarial category, not a real-world
  document set. Baseline numbers here establish that the mechanism works
  and characterize its failure modes; they are not a claim about
  production-scale performance.
- The out-of-corpus refusal failure (0/2, see Interpretation in the
  report) and the ambiguous-term retrieval miss are real, reproducible
  findings from this run, kept as documented limitations (ADR-002,
  PLAN.md roadmap) rather than tuned away by cherry-picking a threshold
  that happens to fix this specific small corpus.
- `expected_answer_contains` keyword matching is a substring heuristic,
  not semantic equivalence — it will under-count correct answers phrased
  with genuine synonyms the corpus text doesn't use, and can't detect a
  fluent-but-wrong answer that happens to repeat the right keywords out of
  context. A real LLM-as-judge pass (mentioned in PLAN.md's evaluation
  section as a planned addition once agent validation is built) would
  catch what this heuristic can't.

## Alternatives considered
- **LLM-as-judge scoring for every metric** — more accurate for
  groundedness and answer quality, but non-deterministic (can't be
  reproduced bit-for-bit in CI without an API key and without run-to-run
  variance) and costs money per CI run. Rejected as the *only* method for
  that reason; kept as a documented future addition rather than replacing
  the deterministic heuristic wholesale.