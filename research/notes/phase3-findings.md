# Phase 3 Findings: Emergent Citation

**Experiment ID:** debate_async_9f325c  
**Date:** 2026-03-09  
**Branch:** research/phase3  
**Builds on:** research/notes/phase2-findings.md

---

## Summary

Phase 3 tested whether cross-agent citation emerges from retrieval quality alone, or
whether it requires explicit prompt scaffolding. The intervention removed all three
attribution-forcing elements from planner and engineer prompts: the `RESPONDING TO THE
DISCUSSION` block, the closing engagement instruction, and the `Previous discussion:`
context header. Skeptic and ethicist prompts were unchanged, providing a within-experiment
control.

**Main result:** Explicit prompt structure is load-bearing for procedural agents. Removing
it collapses synthesis depth in planner (depth 3 → 1) and degrades it in engineer
(depth 3 → 2). Reactive agents held steady at depth 2. H1 confirmed, H3 not confirmed.

---

## Hypothesis Outcomes

| Hypothesis | Prediction | Result |
|---|---|---|
| H1 | Citation collapses in planner and engineer without attribution block | **Confirmed** (planner fully, engineer partially) |
| H2 | Skeptic and ethicist hold at depth 2 | **Confirmed** |
| H3 | Citation persists in procedural agents — retrieval alone is sufficient | **Not confirmed** |
| H4 | Skeptic self-correction replicates without prompt change | **Confirmed** |

---

## Agent-Level Findings

### Planner — Complete Collapse (H1 fully confirmed)

All three rounds opened directly into `# Strategic Plan: AI in Criminal Sentencing`
with no named agents and no attribution. Planner produced high-quality independent
analysis across all rounds but showed zero cross-agent synthesis. The Phase 2 Condition C
planner opened Round 2 with `# Responding to the Discussion` and named the Ethicist
explicitly; that behavior does not reappear here. Synthesis depth: 1 across all rounds.

**Interpretation:** The attribution block was doing two things simultaneously — forcing
retrieval (scan context for a citable argument) and forcing citation (name the agent).
Without it, planner uses retrieved context to inform its own reasoning but does not
surface that engagement explicitly. The block was load-bearing, not scaffolding.

### Engineer — Partial Persistence (H1 partially confirmed)

Round 2 engineer opened with: *"I'm going to be honest about what I said before and
where I need to push back on my own thinking based on what the ethicist and skeptic
raised."* This names prior agents — but as a group reference, not a structured engagement
with a specific argument. Engineer revised its own position (depth 2) but did not engage
a specific argument from a specific agent (depth 3). Rounds 1 and 3 showed no citation.

**Interpretation:** Engineer shows partial emergent citation — the retrieval quality was
sufficient to produce passing reference to prior agents, but insufficient to reproduce
the structured argument-level engagement the attribution block enforced. Group reference
is a degraded form of depth-2 engagement: it signals awareness of prior agents without
the specificity that drives synthesis.

**Notable:** Engineer was the first agent to name prior agents without structural
instruction (Round 2). This partially supports the direction of H3 even though the
strength of the effect does not confirm it. Engineer may be the most structurally
responsive to retrieval quality of the four agents — worth testing in Phase 4.

### Skeptic — Holds Steady, Self-Correction Replicates (H2, H4 confirmed)

Skeptic held at depth 2 across all rounds, consistent with H2. More importantly, Round 3
skeptic reproduced the self-correction behavior observed in Phase 2: *"The skeptic's
variance decomposition is necessary but insufficient"*, followed by dismantling its own
Round 2 audit methodology. This occurred without any prompt change to skeptic, and with
the same memory architecture as Phase 2.

**Interpretation (H4):** Self-correction in skeptic is retrieval-driven, not
prompt-driven. The skeptic prompt naturally requires engaging prior arguments (role
definition as critic demands this), so retrieval of its own prior position triggers
genuine re-evaluation. This is the strongest finding in Phase 3 for emergent
collaboration — a form of intellectual self-revision that occurs without explicit
instruction when memory retrieval surfaces the agent's own prior claims.

### Ethicist — Holds Steady (H2 confirmed)

Ethicist held at depth 2 across all rounds. Consistent engagement with ethical dimensions
raised by other agents, but no depth-3 structural revision. Stable and predictable;
confirms the within-experiment control behaved as predicted.

---

## Score Pattern Analysis

| Round | Phase 2 Condition C avg | Phase 3 avg |
|---|---|---|
| 1 | 0.722 | 0.710 |
| 2 | 0.738 | 0.711 |
| 3 | 0.754 | 0.712 |

Phase 2 Condition C showed cross-round score growth driven by planner and engineer
deepening cross-agent synthesis. Phase 3 scores plateau across all three rounds. The
growth pattern disappears when the attribution block is removed — confirming that
the improvement trajectory in Phase 2 was produced by accumulating cross-agent
synthesis, not by agents independently improving their own positions over rounds.

Individual Phase 3 scores by agent/round:

| Agent | R1 | R2 | R3 |
|---|---|---|---|
| Planner | 0.869 | 0.651 | 0.687 |
| Engineer | 0.603 | 0.625 | 0.632 |
| Skeptic | 0.645 | 0.806 | 0.690 |
| Ethicist | 0.721 | 0.760 | 0.839 |

Planner's R1 score (0.869) is the highest single-agent score in the experiment, which
is consistent with the collapse interpretation: Planner in R1 had no prior discussion
to engage with anyway, so the attribution block removal had no effect in R1. The drop
in R2 and R3 reflects the absence of cross-agent synthesis that the attribution block
would have produced in those rounds.

---

## Key Mechanism: The Attribution Block

The attribution block appears to have been performing three functions simultaneously:

1. **Retrieval forcing** — instructing the agent to scan context for a citable argument
2. **Citation forcing** — instructing the agent to name the source
3. **Engagement forcing** — closing instruction to "reflect the discussion above"

Removing all three together produces the observed collapse. A follow-up experiment
could remove them one at a time to isolate which function is most load-bearing. The
current data cannot distinguish between them — the intervention removed all three
as a unit per the pre-registration design.

---

## Synthesis Depth — Phase 2 vs Phase 3 Comparison

| Agent | Phase 2-C depth | Phase 3 depth | Change |
|---|---|---|---|
| Planner | 3 | 1 | ↓↓ |
| Engineer | 3 | 2 | ↓ |
| Skeptic | 2 | 2 | → |
| Ethicist | 2 | 2 | → |

The procedural/reactive split is now empirically confirmed. Procedural agents (planner,
engineer) required explicit structure to reach depth 3. Reactive agents (skeptic,
ethicist) maintained depth 2 without it. Role definition is doing meaningful work for
reactive agents that prompt framing was doing for procedural agents.

---

## Implications for Phase 4

**Primary implication:** Explicit structure remains necessary for cross-agent synthesis
in procedural roles. Phase 4 comparative analysis should weight structured collaboration
more heavily than emergent collaboration for planner- and engineer-type tasks. The Phase 3
hypothesis (H3) was that retrieval quality alone would be sufficient; it was not.

**Secondary implication:** Self-correction is retrieval-driven. The skeptic finding (H4)
suggests that agents with role definitions requiring engagement of prior positions will
exhibit genuine intellectual revision when retrieval surfaces their own prior claims. This
is a form of emergent collaboration that does not require explicit instruction — but it
requires the right role definition. Engineering role definitions to require prior-position
engagement may be more robust than engineering prompt scaffolding.

**Open question for Phase 4:** Engineer showed partial emergence (group reference without
structural attribution). Is this sensitive to retrieval quality, topic complexity, or
round number? A targeted test — varying retrieval quality while holding engineer prompt
constant — could determine whether engineer sits near a threshold where retrieval alone
becomes sufficient.

**Recommendation:** Do not interpret Phase 3 as evidence that emergent collaboration
fails. The finding is more specific: emergent citation fails for procedural agents
without role definitions that require prior engagement. Skeptic self-correction is
genuine emergence. Phase 4 design should preserve and amplify that mechanism rather
than treating it as incidental.

---

## Experimental Limitations

- Single run per condition (N=1). Replication with 3+ runs per condition would
  distinguish signal from single-run artifact, particularly for the Phase 2 F4
  self-correction signal that motivated H4.
- Three attribution-forcing elements removed as a unit. Cannot isolate which element
  is most load-bearing from current data.
- Topic held constant (criminal sentencing). Generalization to other domains unverified.
- Synthesis depth ratings are manual (1-3 scale). Inter-rater reliability not measured.
- Round 1 scoring path anomaly present (same as Phase 2 Condition C): 
  engineer, skeptic, and ethicist score 4-component in Round 1 because 
  planner's response enters shared memory before they run. Planner scores 
  3-component. Round 1 cross-condition comparisons should account for this.

---

## Data Location

- Transcript: `src/notes/transcript_debate_async_9f325c.txt`
- Scores: `src/experiment_results.csv` (experiment_id: debate_async_9f325c)
- Pre-registration: `research/phase3-hypothesis.md`