"""
Condition B: Context-only control for Phase 2 memory experiment.

Uses sliding window context (no AgentMemory) to isolate whether
memory retrieval contributes beyond simply having prior context
injected into the prompt.

Run BEFORE run_memory.py. Same topic, same CSV file, different
context_window label in results.

Hypothesis: If condition B scores similarly to condition C (memory),
retrieval adds no value over a well-sized sliding window. If C > B
in rounds 2-3, retrieval is doing real work.
"""

import asyncio
import sys
import time
import uuid
import logging
from experiment_logger import log_result
from scorer import score_response
from config import config
from llm_client import call_llm
from prompts import build_messages

logger = logging.getLogger(__name__)

EXECUTION_MODE = "sequential"
RESPONSE_TRUNCATE_CHARS = 500
PHASE = "async_debate"
CONTEXT_WINDOW_SIZE = 2


def build_context(round_history: list[dict], window: int) -> tuple[str, str]:
    """
    Sliding window context — no retrieval scoring, no AgentMemory.
    Injects the last `window` rounds of all agents' responses verbatim.
    This is the attribution control: same information volume as memory,
    but selected by recency alone (no relevance or importance weighting).
    """
    if not round_history:
        return "", "round_1_no_context"

    rounds_present = sorted(set(e["round"] for e in round_history))
    rounds_in_window = rounds_present[-window:]
    entries = [e for e in round_history if e["round"] in rounds_in_window]

    context = ""
    for e in entries:
        truncated = e["response"][:RESPONSE_TRUNCATE_CHARS]
        if len(e["response"]) > RESPONSE_TRUNCATE_CHARS:
            truncated += "... [truncated]"
        context += f"Round {e['round']} - {e['agent']}: {truncated}\n\n"

    label = f"control_window_r{rounds_in_window[0]}-{rounds_in_window[-1]}"
    return context, label


async def process_agent(
    agent_name: str,
    topic: str,
    context: str,
    round_num: int,
    experiment_id: str,
    filename: str,
    all_responses: list,
    context_label: str,
) -> None:
    messages = build_messages(agent_name, topic, context)
    text, usage = await call_llm(messages, config)
    logger.debug("[%s] eval_count=%d", agent_name, usage["eval_count"])
    response = f"[{agent_name.capitalize()}]\n{text}"

    transcript_file = f"transcript_{experiment_id}.txt"
    with open(transcript_file, "a") as f:
        f.write(f"=== Round {round_num} | {agent_name} ===\n")
        f.write(response)
        f.write("\n\n")

    quality_score, scoring_path = score_response(response, topic, context)
    log_result(
        experiment_id=experiment_id,
        agent=agent_name,
        round_num=round_num,
        quality_score=quality_score,
        phase=PHASE,
        scoring_path=scoring_path,
        context_window=context_label,
        execution_mode=EXECUTION_MODE,
        filename=filename,
    )
    all_responses.append({
        "round": round_num,
        "agent": agent_name,
        "response": response,
    })
    print(f"  ✓ {agent_name}: {response[:60]}... (score: {quality_score:.3f}, path: {scoring_path})")


async def run_control(
    topic: str,
    rounds: int = 3,
    filename: str = "experiment_results.csv",
) -> tuple[str, list[dict]]:
    start_time = time.perf_counter()
    experiment_id = f"debate_control_{uuid.uuid4().hex[:6]}"

    agents = ["planner", "engineer", "skeptic", "ethicist"]
    all_responses = []

    print("=" * 70)
    print(f"CONTROL RUN (no memory): '{topic}'")
    print(f"Rounds: {rounds} | Agents: {len(agents)} | "
          f"Window: {CONTEXT_WINDOW_SIZE} | Mode: {EXECUTION_MODE}")
    print("=" * 70)

    for round_num in range(1, rounds + 1):
        round_start = time.perf_counter()
        print(f"\n--- Round {round_num} (at {round_start - start_time:.2f}s) ---")

        context, context_label = build_context(all_responses, CONTEXT_WINDOW_SIZE)

        for name in agents:
            try:
                await process_agent(
                    name, topic, context, round_num,
                    experiment_id, filename, all_responses, context_label,
                )
            except Exception as e:
                logger.error("[%s] failed in round %d: %s", name, round_num, e)

        round_elapsed = time.perf_counter() - round_start
        print(f"--- Round {round_num} completed in {round_elapsed:.2f}s ---")

    total_time = time.perf_counter() - start_time
    print("\n" + "=" * 70)
    print("CONTROL RUN COMPLETE")
    print(f"Total elapsed: {total_time:.2f}s | Experiment ID: {experiment_id}")
    print("=" * 70)

    return experiment_id, all_responses


if __name__ == "__main__":
    topic = sys.argv[1] if len(sys.argv) > 1 else \
        "Should AI be used in criminal sentencing?"
    asyncio.run(run_control(topic=topic))