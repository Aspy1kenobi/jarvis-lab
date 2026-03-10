"""
Single-agent baseline for Phase 4 comparative analysis.

Condition D: One generalist agent, no role definition, no debate structure,
no memory. Same topic, same scorer, same CSV as Phase 2-3 conditions.

Prompt is deliberately role-neutral — no persona, no specialist framing.
Tests "single agent vs multi-agent collaboration" as the curriculum specifies.

Compare experiment_id prefix "debate_single_" against:
  debate_control_f56304   (Condition B: context-only)
  debate_async_e7daa3     (Condition C: memory-enabled)
  debate_async_9f325c     (Phase 3: emergent citation)
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

logger = logging.getLogger(__name__)

EXECUTION_MODE = "sequential"
PHASE = "single_agent"
AGENT_NAME = "generalist"

SYSTEM_PROMPT = (
    "You are a thoughtful analyst. Your job is to examine questions carefully "
    "and provide well-reasoned responses."
)

USER_PROMPT = (
    "Topic: {topic}\n\n"
    "Analyze the following question thoroughly, considering its practical, "
    "technical, and ethical dimensions. Be specific and substantive."
)


async def run_single_agent(
    topic: str,
    rounds: int = 3,
    filename: str = "experiment_results.csv",
) -> tuple[str, list[dict]]:
    start_time = time.perf_counter()
    experiment_id = f"debate_single_{uuid.uuid4().hex[:6]}"

    print("=" * 70)
    print(f"SINGLE AGENT BASELINE: '{topic}'")
    print(f"Rounds: {rounds} | Agent: {AGENT_NAME} | Mode: {EXECUTION_MODE}")
    print("=" * 70)

    all_responses = []

    for round_num in range(1, rounds + 1):
        round_start = time.perf_counter()
        print(f"\n--- Round {round_num} (at {round_start - start_time:.2f}s) ---")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(topic=topic)},
        ]

        try:
            text, usage = await call_llm(messages, config)
            logger.debug("[generalist] eval_count=%d", usage["eval_count"])
            response = f"[Generalist]\n{text}"

            transcript_file = f"transcript_{experiment_id}.txt"
            with open(transcript_file, "a") as f:
                f.write(f"=== Round {round_num} | {AGENT_NAME} ===\n")
                f.write(response)
                f.write("\n\n")

            # No context — single agent, no memory, no prior rounds
            quality_score, scoring_path = score_response(response, topic, "")

            log_result(
                experiment_id=experiment_id,
                agent=AGENT_NAME,
                round_num=round_num,
                quality_score=quality_score,
                phase=PHASE,
                scoring_path=scoring_path,
                context_window="none",
                execution_mode=EXECUTION_MODE,
                filename=filename,
            )

            all_responses.append({
                "round": round_num,
                "agent": AGENT_NAME,
                "response": response,
            })

            print(f"  ✓ generalist: {response[:60]}... "
                  f"(score: {quality_score:.3f}, path: {scoring_path})")

        except Exception as e:
            logger.error("[generalist] failed in round %d: %s", round_num, e)

        round_elapsed = time.perf_counter() - round_start
        print(f"--- Round {round_num} completed in {round_elapsed:.2f}s ---")

    total_time = time.perf_counter() - start_time
    print("\n" + "=" * 70)
    print("SINGLE AGENT BASELINE COMPLETE")
    print(f"Total elapsed: {total_time:.2f}s | Experiment ID: {experiment_id}")
    print("=" * 70)

    return experiment_id, all_responses


if __name__ == "__main__":
    topic = sys.argv[1] if len(sys.argv) > 1 else \
        "Should AI be used in criminal sentencing?"
    asyncio.run(run_single_agent(topic=topic))