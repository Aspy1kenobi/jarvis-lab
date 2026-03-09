# JARVIS Research — Phase 4 Kickoff: Workshop Paper

## Context for This Session
I'm a self-directed AI researcher writing a workshop paper based on a 
completed multi-agent debate framework called JARVIS. Phase 4 is the 
writing and submission phase. All experimental work is complete.

**Target venue:** ICML 2026 workshop (submission window ~May 2026)
**Fallback / follow-on:** NeurIPS 2026 workshop
**Paper type:** Workshop paper (not position paper — that's the follow-on 
once more replication exists)

---

## The Research Program (3 phases complete)

**Primary research question:** How can multiple AI agents collaborate to 
produce better outcomes than a single agent, and what collaboration 
structures (explicit vs emergent) are most effective?

**Phase 1 — Structured debate (complete)**
Four-agent debate loop: planner, engineer, skeptic, ethicist.
Key finding: Procedural agents (planner, engineer) templated responses 
across rounds. Reactive agents (skeptic, ethicist) iterated. Diagnosed 
as a prompt problem, not a model capability problem. Scorer inversion 
artifact identified and fixed.

**Phase 2 — Memory integration (complete)**
Replaced sliding window context with AgentMemory — ephemeral per-agent 
memory implementing Generative Agents retrieval formula 
(recency + importance + relevance). Three conditions:
- Condition A: Phase 1 baseline (git tag: debate_async_c18f20)
- Condition B: Context-only control (experiment: debate_control_f56304)
- Condition C: Memory-enabled (experiment: debate_async_e7daa3)

Key findings:
- Memory advantage real and grows across rounds 
  (+0.011 R1, +0.161 R2, +0.120 R3 vs control)
- First confirmed synthesis signal: planner named Ethicist's 
  contestability argument, collapsed "constrained use" as a category, 
  redefined plan goal
- Engineer benefited most from memory (+0.20 R2, +0.13 R3)
- Round 1 scoring path anomaly: sequential execution means agents 2-4 
  get cross-agent context in R1; only agent 1 gets true baseline

**Phase 3 — Emergent citation (complete)**
Removed all three attribution-forcing elements from planner and engineer 
prompts: RESPONDING TO THE DISCUSSION block, closing engagement 
instruction, "Previous discussion:" header → replaced with "Context:". 
Skeptic and ethicist prompts unchanged (within-experiment control).
Experiment: debate_async_9f325c

Hypothesis outcomes:
- H1 confirmed: Citation collapses in planner (depth 3→1), 
  degrades in engineer (depth 3→2)
- H2 confirmed: Skeptic and ethicist hold at depth 2
- H3 not confirmed: Retrieval alone insufficient for procedural agents
- H4 confirmed: Skeptic self-correction replicates without prompt change

**Primary finding (for abstract):**
Emergent self-correction observed in skeptic agent across two independent 
runs. Agent retrieved its own prior position, compared against new 
retrieval material, and revised — without explicit instruction. Behavior 
absent in Phase 1 baseline, not explained by prompt changes. Memory 
retrieval is the candidate mechanism; controlled isolation is future work.

**Procedural/evaluative distinction (empirically confirmed):**
Procedural agents (planner, engineer) require explicit scaffolding to 
achieve cross-agent synthesis. Evaluative agents (skeptic, ethicist) 
maintain engagement without it — role definitions requiring judgment 
about prior claims make retrieval-driven behavior sufficient.

---

## Current Abstract Draft (v1)

We investigate whether multi-agent AI collaboration can produce emergent 
reasoning behaviors beyond what explicit prompt scaffolding produces. 
Using a four-agent debate framework (planner, engineer, skeptic, ethicist) 
with an episodic memory architecture implementing the Generative Agents 
retrieval formula, we conduct a sequential research program across three 
phases: a sliding window context baseline, a memory-enabled structured 
condition, and a memory-enabled condition with attribution scaffolding 
selectively removed. Hypotheses for each phase were pre-registered before 
experimental runs.

Across phases, we find that explicit prompt structure is load-bearing for 
cross-agent synthesis in procedural roles: removing attribution 
requirements collapses synthesis depth in planner agents and degrades it 
in engineer agents. Evaluative agents (skeptic, ethicist) maintain 
cross-agent engagement without explicit instruction, consistent with role 
definitions that inherently require judgment about prior claims.

The primary finding is a replicating case of emergent self-correction in 
the skeptic agent. Across two independent experimental runs with differing 
prompt conditions, the skeptic agent revised its own prior arguments in 
round 3 when retrieved context surfaced its round 2 position alongside 
new material. This behavior was absent in the Phase 1 baseline, was not 
produced by prompt changes, and is consistent with memory retrieval as 
the enabling mechanism — though controlled isolation of that mechanism 
remains for future work.

We find that the answer to whether emergent collaboration is achievable 
is role-dependent: emergent collaboration is observed in evaluative 
agents, while procedural agents require explicit scaffolding to achieve 
equivalent synthesis depth. This distinction between prompt-forced 
citation and retrieval-driven self-correction has implications for the 
design of multi-agent systems where reasoning quality, not just output 
diversity, is the target.

---

## Paper Structure (agreed, not yet drafted)

Standard workshop paper format, 6-8 pages:
1. Introduction
2. Related Work
3. System Architecture
4. Experimental Design
5. Results
6. Discussion
7. Limitations and Future Work

---

## Writing Workflow
- Draft in Markdown by section (paper/drafts/)
- Convert to LaTeX for submission (paper/latex/)
- Apply ICML 2026 style file
- BibTeX references built as we go (paper/refs.bib)

**Branch:** research/phase4 (to be created)
**Directory:** paper/drafts/, paper/latex/, paper/refs.bib

---

## Core References (already read and annotated)
- Vaswani et al. (2017) — Attention Is All You Need
- Bai et al. (2022) — Constitutional AI (Anthropic)
- Park et al. (2023) — Generative Agents
- Yao et al. (2022) — ReAct

Additional references needed:
- Multi-agent debate literature (Du et al. 2023 "Improving Factuality 
  and Reasoning through Multiagent Debate" is the key one)
- Prompt engineering / chain-of-thought literature
- Any prior work on emergent behavior in LLM systems

---

## Key Decisions Already Made
- Workshop paper in May, position paper as follow-on after replication
- "Evaluative" not "reactive" for skeptic/ethicist agent type
- Primary finding framed as "we observe a case of" not 
  "we demonstrate that"
- Pre-registration discipline maintained across all phases — 
  this should be highlighted as methodological rigor in the paper
- Limitations stated directly: N=2 on primary finding, single topic, 
  single rater on synthesis depth ratings

---

## To Start
Create research/phase4 branch, set up paper/ directory structure, 
then begin with the outline — agree on what each section needs to 
accomplish before drafting any prose.

Paste this kickoff, confirm branch, and we start with the outline.