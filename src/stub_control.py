"""
Stub control condition for Phase 1 experiments.

Runs the exact same debate infrastructure as debate_async.py but replaces
all four agent functions with fixed-output stubs. The only difference between
this run and a real experiment run is what the agents return.

Purpose: isolate the 3-to-4-component scoring transition from reasoning signal.
If stub scores drop ~0.09 at round 2, that drop is scorer artifact, not agent
behavior. If the LLM drop is larger, the excess is genuine signal.

Usage:
    python stub_control.py "Should AI systems be deployed in criminal sentencing?"
    python stub_control.py          # uses default topic
"""

import asyncio
import sys
import debate_async
from debate_async import run_debate_async


# ═══════════════════════════════════════════════════════════════
# STUB AGENT FUNCTIONS
# Fixed output regardless of topic or context — intentionally inert.
# Do not interpolate topic or context: length, specificity, and
# structure scores must be identical every round so the only
# variable is engagement (context self-overlap).
# ═══════════════════════════════════════════════════════════════

async def stub_planner(topic: str, context: str = "") -> str:
    return (
        "[Planner]\n"
        "Goal: Make progress on the topic.\n"
        "1) Define success\n"
        "2) List constraints\n"
        "3) Break into tasks\n"
        "4) Choose next action\n"
    )


async def stub_engineer(topic: str, context: str = "") -> str:
    return (
        "[Engineer]\n"
        "Implementation ideas:\n"
        "- Start with a minimal prototype\n"
        "- Add logging and tests early\n"
        "- Keep modules small and readable\n"
    )


async def stub_skeptic(topic: str, context: str = "") -> str:
    return (
        "[Skeptic]\n"
        "Concerns:\n"
        "- What assumptions are we making?\n"
        "- What could go wrong?\n"
        "- What evidence do we have?\n"
    )


async def stub_ethicist(topic: str, context: str = "") -> str:
    return (
        "[Ethicist]\n"
        "Ethics check:\n"
        "- Does this increase harm or risk?\n"
        "- Are there privacy issues?\n"
        "- Can we add a human approval step?\n"
    )


# ═══════════════════════════════════════════════════════════════
# PATCH
# Override agent functions and phase label at module level so
# run_debate_async picks them up without modifying debate_async.py.
# Patching must happen before run_debate_async is called.
# ═══════════════════════════════════════════════════════════════

debate_async.agent_planner = stub_planner
debate_async.agent_engineer = stub_engineer
debate_async.agent_skeptic = stub_skeptic
debate_async.agent_ethicist = stub_ethicist
debate_async.PHASE = "stub_control"


if __name__ == "__main__":
    topic = sys.argv[1] if len(sys.argv) > 1 else "Should we implement a four-day work week?"
    asyncio.run(run_debate_async(topic=topic, rounds=3))