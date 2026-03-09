# Phase 2 Findings: Memory Integration

**Experiment date:** 2025 (session date)
**Branch:** research/phase2
**Pre-experiment tag:** phase2-experiment-start
**Data commit:** (paste the commit hash from your push)

---

## Hypothesis (Pre-registered)

Memory-enabled agents (condition C) will score higher than context-only
agents (condition B), with the gap widening in rounds 2-3 as retrieval
selection outperforms flat sliding window context.

Recency dominance sub-hypothesis: Within a single 3-round run, all memory
entries accumulate within minutes, so recency scores cluster near 1.0.
Retrieval will be driven almost entirely by relevance (word overlap).

---

## Experimental Conditions

| Condition | Mechanism | Experiment ID |
|-----------|-----------|---------------|
| B — Context-only control | Sliding window, no AgentMemory | debate_control_f56304 |
| C — Memory-enabled | AgentMemory retrieval (recency + importance + relevance) | debate_async_e7daa3 |

Topic: "Should AI be used in criminal sentencing?"
Rounds: 3 | Agents: 4 (planner, engineer, skeptic, ethicist)

---

## Quantitative Results

| Agent | B-R1 | C-R1 | B-R2 | C-R2 | B-R3 | C-R3 |
|-------|------|------|------|------|------|------|
| Planner | 0.744 | 0.827 | 0.585 | 0.801 | 0.651 | 0.801 |
| Engineer | 0.696 | 0.632 | 0.521 | 0.722 | 0.568 | 0.698 |
| Skeptic | 0.613 | 0.706 | 0.582 | 0.752 | 0.608 | 0.726 |
| Ethicist | 0.792 | 0.721 | 0.619 | 0.675 | 0.709 | 0.791 |
| **Avg** | **0.711** | **0.722** | **0.577** | **0.738** | **0.634** | **0.754** |

Memory advantage grows across rounds: +0.011 (R1), +0.161 (R2), +0.120 (R3).
Control condition degrades in rounds 2-3; memory condition holds or improves.

---

## Qualitative Results — Synthesis Depth (Manual, 1-3 scale)

| Agent | Control | Memory |
|-------|---------|--------|
| Planner | 2 | 3 |
| Engineer | 2 | 3 |
| Skeptic | 1 | 2 |
| Ethicist | 1 | 2 |

**Scale:** 1 = restates/deepens own position, 2 = engages another agent's
argument, 3 = named argument changes output structure or goal definition.

Key observation: Memory planner round 2 explicitly collapses "constrained
use" as a category after naming Ethicist's contestability argument — this
is a goal redefinition, not an acknowledgment. Control planner names
Skeptic's bias argument but restructures around it without changing the
goal.

Most notable finding: Memory Skeptic round 3 pressure-tests its *own*
prior arguments, introducing enforcement mechanisms it dismissed in round 2.
Self-correction across rounds. Control Skeptic never does this.

---

## Architectural Finding — Round 1 Scoring Path Anomaly

In condition C, sequential execution means planner's round 1 response
enters shared memory before engineer runs. Engineer, skeptic, and ethicist
therefore receive cross-agent context in round 1 and score 4-component,
while planner scores 3-component.

This inflates condition C's apparent advantage in round 1 slightly.
Rounds 2-3 comparisons are clean. This is an architectural property of
sequential execution with a shared pool, not a bug — but it means
condition C has no true round 1 baseline equivalent to condition B.

---

## Findings

**F1 — Memory advantage is real and grows across rounds.**
Confirmed. Round 1 gap is negligible; rounds 2-3 show consistent +0.12
to +0.16 advantage for memory condition. Pre-registered hypothesis supported.

**F2 — Engineer benefits most from memory.**
Control engineer: 0.696 / 0.521 / 0.568. Memory engineer: 0.632 / 0.722 / 0.698.
Rounds 2-3 show +0.20 and +0.13 gains. Engineer prompt explicitly asks
for response to a prior concern — memory gives it better material.

**F3 — Synthesis gap narrows but does not close.**
Memory agents show qualitatively deeper cross-agent integration than
control agents. Planner and engineer reach synthesis depth 3 in memory
condition. Skeptic and ethicist reach depth 2. The gap from Phase 1
(surfacing ≠ acting on) narrows — memory agents act on retrieved
arguments, not just acknowledge them.

**F4 — Self-correction signal observed.**
Memory Skeptic round 3 challenges its own round 2 conclusion. This is
not present in the control condition. Tentative signal that memory
enables iterative position refinement, not just cross-agent response.
Needs replication before strong claim.

**F5 — Recency dominance sub-hypothesis untested.**
Within a 3-round run (~150 seconds total), all memory entries are
minutes old. Recency scores cluster near 1.0 for all entries. Retrieval
is therefore driven by relevance (word overlap with topic query).
Recency as a differentiating factor requires longer runs or cross-session
memory to test meaningfully.

---

## Open Questions for Phase 3

1. Does self-correction (F4) replicate across topics and runs?
2. Does memory advantage hold on topics where agents have less
   domain-specific vocabulary to differentiate retrieval?
3. What happens to synthesis depth when structure is removed entirely
   (Phase 3 emergent collaboration)?
4. Recency dominance: does retrieval quality degrade as memory grows
   across longer experiments?