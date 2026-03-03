# JARVIS — Research Environment README

> **Note:** This README is for the research environment. The education environment (jarvis-lab/) is separate. Do not mix them.

---

## What This Is

JARVIS is a multi-agent research framework built to answer one question:

**"How can multiple AI agents collaborate to produce better outcomes than a single agent, and what collaboration structures (explicit vs emergent) are most effective?"**

This is not a chatbot, assistant, or product. It is a research instrument.

---

## Research Phases

| Phase | Name | Status | Description |
|-------|------|--------|-------------|
| 1 | Structured Debate | Infrastructure complete | Constitutional AI-inspired debate loop with sequential and async modes |
| 2 | Memory Integration | Planned | Generative Agents-inspired retrieval scoring wired to live LLM |
| 3 | Emergent Collaboration | Planned | Agents without explicit structure; parallel vs sequential comparison |
| 4 | Comparative Analysis | Planned | Cross-phase analysis, human evaluation layer, publication prep |

---

## Running an Experiment

### Prerequisites

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your API key
```

### Run a debate (sequential)

```bash
cd src
python run_experiment.py
```

### Run a debate (async, faster)

```bash
cd src
python debate_async.py
```

### Run from Python

```python
from run_experiment import run_experiment
run_experiment("Your debate topic here", rounds=3)
```

---

## Project Structure

```
jarvis-lab/
├── src/
│   ├── memory.py           # Memory class with NumPy retrieval scoring
│   ├── agents.py           # Five agent functions (sync)
│   ├── debate.py           # Sequential debate loop
│   ├── debate_async.py     # Async debate loop (asyncio.gather per round)
│   ├── scorer.py           # Three-component response scorer
│   ├── run_experiment.py   # End-to-end pipeline entry point
│   ├── experiment_logger.py# CSV append logger
│   ├── experiment_plots.py # Quality curve and participation bar chart
│   ├── main.py             # Interactive CLI (JARVIS terminal)
│   └── research_mode.py    # Research-specific CLI commands
├── data/
│   └── memory.json         # Persistent note storage
├── results/
│   ├── quality_curve.png
│   └── agent_participation.png
├── experiment_results.csv  # All experiment logs
├── .env                    # API keys — never commit this
├── .env.example            # Template — safe to commit
└── .gitignore
```

---

## Agents

| Agent | Role | Behavior |
|-------|------|----------|
| Planner | Strategic | Defines success, breaks into tasks, identifies next action |
| Engineer | Technical | Focuses on implementation, prototyping, architecture |
| Skeptic | Critical | Identifies assumptions, failure modes, verification gaps |
| Ethicist | Ethical | Examines harm, privacy, human oversight |
| Imagination | Creative | LLM-powered; generates novel framings (requires model) |

Agents accept `(topic: str, context: str = "")`. Context accumulates across rounds.

---

## Scoring

Responses are scored on three normalized components (each 0.0–1.0, averaged):

- **Length:** Word count relative to 200-word ceiling
- **Specificity:** Topic word coverage (set intersection, not repetition count)
- **Structure:** Presence of reasoning keywords via stem matching

**Known limitation:** Structure keywords are biased toward Skeptic's vocabulary. Scores measure pipeline integrity with deterministic agents. Meaningful differentiation requires live LLM integration (Phase 2).

---

## Experiment Logging

Every agent response is logged to `experiment_results.csv` with:

```
experiment_id, agent, round, quality_score, phase, timestamp
```

Filter by `experiment_id` to isolate a single run. Each run generates a unique ID: `debate_<hex>`.

---

## Tagging Convention

```bash
# Phase boundaries
git tag -a v0.3-pre-phase1 -m "Description"

# Experiment checkpoints (before structured runs)
git tag -a exp/phase1/debate-001 -m "Hypothesis: X. Conditions: Y."

# Surprising results (after exploratory runs)
git tag -a exp/phase3/emergent-007 -m "Skeptic referenced planner unprompted at round 4"

# Push tags explicitly
git push origin --tags
```

---

## Active Hypotheses

**Alpha hypothesis:** Multi-agent systems increase reasoning surface area, approximated by: distinct considerations, conditional branches, counterarguments raised, stakeholder perspectives identified.

**Sequential vs parallel:** Sequential debates may appear higher quality due to agent ordering effects (planner always frames first). Phase 4 analysis should control for or randomize agent order.

---

## Foundational Papers

- Vaswani et al. — *Attention Is All You Need* (transformer architecture)
- Bai et al. — *Constitutional AI* (structured critique and revision; basis for Phase 1)
- Park et al. — *Generative Agents* (memory retrieval scoring; basis for Phase 2)
- Yao et al. — *ReAct* (reasoning + acting; basis for Phase 3 architecture)

---

## What's Not Here Yet

- Live LLM integration (Phase 2)
- `.env` / API key configuration
- Human evaluation layer (Phase 4)
- Jupyter notebook for interactive analysis (Phase 4)
- Web interface (post-Phase 4, if needed for sharing)
