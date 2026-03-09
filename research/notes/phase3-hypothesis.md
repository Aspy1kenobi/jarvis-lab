# Phase 3 Hypothesis: Emergent Citation

**Pre-registration date:** 2026-03-09
**Branch:** research/phase3
**Builds on:** Phase 2 findings (research/notes/phase2-findings.md)

---

## Research Question

Is citation behavior in multi-agent debate prompt-dependent, or does it
emerge naturally when agents have sufficient retrieval material?

---

## Experimental Design

**Intervention:** Remove the RESPONDING TO THE DISCUSSION attribution
block from planner and engineer prompts. Also remove closing instructions
to "reflect the discussion above" and change context header from
"Previous discussion:" to "Context:" — all three explicit engagement
framing elements removed together.

**Control within experiment:** Skeptic and ethicist prompts unchanged.
They have no attribution block and no engagement closing line. Their
citation behavior in Phase 2 was not structurally forced. If citation
persists in skeptic and ethicist but collapses in planner and engineer,
that isolates the attribution block as the causal mechanism.

**Memory architecture:** Unchanged from Phase 2 condition C. Agents
receive retrieved context via AgentMemory (recency + importance +
relevance). The retrieval quality that produced synthesis in Phase 2
is held constant. Only the prompt framing changes.

**Topic:** Same as Phase 2 — "Should AI be used in criminal sentencing?"
Keeps content constant so behavioral differences are attributable to
prompt structure, not topic.

**Runs:** 3 rounds, 4 agents, sequential execution.

---

## Pre-Registered Hypotheses

**H1 (Primary):** Citation rate collapses in planner and engineer
without the attribution block. Agents will use retrieved context to
inform their responses but will not explicitly name or engage with
prior agents' arguments. Synthesis depth regresses to 1-2 for
procedural agents.

**H2 (Control check):** Citation behavior persists in skeptic and
ethicist at approximately Phase 2 levels. Their role definitions
naturally require engaging prior arguments regardless of explicit
instruction. Synthesis depth holds at 2 for reactive agents.

**H3 (Interesting null):** Citation persists in planner and engineer
without instruction. If observed, this would indicate that retrieval
quality alone is sufficient to drive citation behavior — the attribution
block was redundant, not load-bearing. This would be the strongest
possible finding for emergent collaboration.

**H4 (Self-correction replication):** The memory Skeptic round 3
self-correction observed in Phase 2 (F4) was tentative. If it reappears
in Phase 3 without any prompt change to skeptic, that strengthens the
claim that self-correction is retrieval-driven. If it disappears, the
Phase 2 signal may have been a single-run artifact.

---

## Null Prediction

All agents regress to depth-1 behavior — restating and deepening their
own positions without cross-agent engagement. This would indicate that
Phase 2 synthesis was entirely prompt-driven, and that memory retrieval
alone does not produce collaboration without structural scaffolding.

---

## Measurements

**Primary:** Cross-agent citation rate — count of explicit agent names
per response per round. Direct behavioral test of H1 and H2.

**Secondary:** Synthesis depth — same 1-3 manual scale as Phase 2.
  1 = restates or deepens own position
  2 = engages another agent's argument
  3 = named argument changes output structure or goal definition

**New:** First unprompted citation — which agent, which round, produces
the first explicit cross-agent reference. If always skeptic or ethicist,
confirms reactive/procedural split. If planner or engineer, supports H3.

**Quantitative:** Quality scores per agent per round, logged to
experiment_results.csv. Secondary here — behavioral change is the
primary outcome, not score change.

---

## Candidate Explanations If H1 Is Confirmed
(Citation collapses in procedural agents)

The attribution block was doing two things simultaneously: forcing
retrieval (scan context for a citable argument) and forcing citation
(name the agent). Removing it removes both. The self-correction signal
in Phase 2 may have been a downstream effect of forced retrieval
triggering genuine position revision — not prompt-independent reasoning.

## Candidate Explanations If H3 Is Confirmed
(Citation persists without instruction)

Retrieval quality is sufficient. Agents with good context material
naturally reference its source because doing so strengthens their
argument. The attribution block was scaffolding that became unnecessary
once memory provided high-quality, agent-labeled retrieval content.
This would support the Phase 3 goal of emergent collaboration.

---

## Relationship to Primary Research Question

"How can multiple AI agents collaborate to produce better outcomes than
a single agent, and what collaboration structures (explicit vs emergent)
are most effective?"

Phase 3 Option B is the first direct test of whether collaboration
structure can be emergent rather than explicit. A positive finding
(H3 confirmed) would mean the memory architecture itself is sufficient
to produce collaborative behavior — structure emerges from retrieval,
not from instruction. A negative finding (H1 confirmed) would mean
explicit structure remains necessary, and Phase 4 comparative analysis
should weight structured collaboration more heavily.

Either outcome is informative. Pre-registering both prevents
post-hoc interpretation.