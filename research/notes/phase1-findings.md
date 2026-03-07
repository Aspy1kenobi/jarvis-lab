# Phase 1 Findings — experiment debate_async_c18f20

## Alpha Hypothesis Assessment
Weak positive signal, but mechanism is wrong.
- Surface area gain confirmed: full debate > any single response
- Mechanism: specialization (4 domain lenses), NOT cross-agent synthesis
- Cross-agent engagement occurred but was shallow — premise-level, not argument-level

## Agent Persona Finding
- Planner + Engineer: templated across rounds, near-verbatim structure repetition
- Skeptic + Ethicist: genuine round-over-round development
- Explanation: procedural roles template; reactive roles iterate
- Score asymmetry in round 3 recovery reflects this distinction

## Scorer Behavior
- Stub control confirms engagement component rewards self-repetition (fixed vocab = high overlap)
- Engagement penalizes exploration — agents introducing new material score lower
- 3-to-4-component transition artifact: ~0.10-0.11 drop at round 2 regardless of reasoning quality
- Ethicist consistently highest scorer — reactive + structured output format

## Open Questions for Phase 2
- Does memory change templating behavior in procedural agents?
- Can engagement metric be redesigned to reward novel-but-contextual arguments?
- Control for ordering: planner always sets frame — randomize agent order in Phase 4
## The Real Research Question (identified post-experiment)
The ethicist raised that an MVP approach might be ethically impermissible —
not just risky, but categorically inappropriate. No other agent responded.
In a human debate that point forces a response: defend the framing or revise it.
Here it sat unacknowledged.

This is the gap: between surfacing a consideration and the system doing 
something with it. Phase 1 tests whether agents surface more considerations
than a single agent. They do. But surfacing is not collaboration.

The Phase 3 question is not "do agents coordinate without explicit structure?"
It's "under what conditions does a surfaced consideration actually change 
another agent's reasoning?" That's the difference between parallelized 
perspectives and genuine emergence.

## Phase 3 Prerequisites (revised based on findings)
1. Fix planner and engineer prompts — verify round-over-round responsiveness
   before removing structure. Can't test emergence with stateless agents.
2. 2-round responsiveness test: did round 2 output change meaningfully from
   round 1? Direct inspection, not just scores.
3. Phase 3 is only worth running if all four agents are actually in the room.
