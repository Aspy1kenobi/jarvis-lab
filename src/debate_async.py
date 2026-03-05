import asyncio
import time
import uuid
import logging
import functools
from experiment_logger import log_result
from scorer import score_response
from config import config
from llm_client import call_ollama
from prompts import build_messages

logger = logging.getLogger(__name__)


def log_call(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        logger.debug("[%s] called with: %s %s", func.__name__, args, kwargs)
        return await func(*args, **kwargs)
    return wrapper


# ═══════════════════════════════════════════════════════════════
# ASYNC AGENT FUNCTIONS
# ═══════════════════════════════════════════════════════════════

@log_call
async def agent_planner(topic: str, context: str = "") -> str:
    messages = build_messages("planner", topic, context)
    text, usage = await call_ollama(messages, config)
    logger.debug("[planner] eval_count=%d", usage["eval_count"])
    return f"[Planner]\n{text}"


@log_call
async def agent_engineer(topic: str, context: str = "") -> str:
    messages = build_messages("engineer", topic, context)
    text, usage = await call_ollama(messages, config)
    logger.debug("[engineer] eval_count=%d", usage["eval_count"])
    return f"[Engineer]\n{text}"


@log_call
async def agent_skeptic(topic: str, context: str = "") -> str:
    messages = build_messages("skeptic", topic, context)
    text, usage = await call_ollama(messages, config)
    logger.debug("[skeptic] eval_count=%d", usage["eval_count"])
    return f"[Skeptic]\n{text}"


@log_call
async def agent_ethicist(topic: str, context: str = "") -> str:
    messages = build_messages("ethicist", topic, context)
    text, usage = await call_ollama(messages, config)
    logger.debug("[ethicist] eval_count=%d", usage["eval_count"])
    return f"[Ethicist]\n{text}"


@log_call
async def agent_imagination(topic: str, context: str = "") -> str:
    await asyncio.sleep(1)
    from llm_bridge import generate_text, is_available, initialize_llm
    if not is_available():
        return "[Imagination Agent]\nLLM not available"
    initialize_llm()
    text = generate_text(f"Creative ideas about {topic}: ", length=200, temperature=0.8)
    if text:
        return f"[Imagination Agent - Powered by TinyLM]\n{text}"
    return "[Imagination Agent]\nFailed to generate ideas"


# ═══════════════════════════════════════════════════════════════
# PROCESS HELPER
# NOTE: all_responses is a shared list mutated by concurrent tasks.
# This is safe in asyncio (single-threaded event loop) but would
# require a lock if moved to ThreadPoolExecutor or ProcessPoolExecutor.
# ═══════════════════════════════════════════════════════════════

async def process_agent(
    agent_name: str,
    agent_func,
    topic: str,
    context: str,
    round_num: int,
    experiment_id: str,
    filename: str,
    all_responses: list,
) -> None:
    response = await agent_func(topic, context)
    quality_score = score_response(response, topic)
    log_result(
        experiment_id=experiment_id,
        agent=agent_name,
        round_num=round_num,
        quality_score=quality_score,
        phase="async_debate",
        filename=filename,
    )
    all_responses.append({
        "round": round_num,
        "agent": agent_name,
        "response": response,
    })
    print(f"  ✓ {agent_name}: {response[:60]}... (score: {quality_score:.3f})")


# ═══════════════════════════════════════════════════════════════
# MAIN ASYNC DEBATE LOOP
# ═══════════════════════════════════════════════════════════════

async def run_debate_async(
    topic: str,
    rounds: int = 3,
    filename: str = "experiment_results.csv",
) -> tuple[str, list[dict]]:
    start_time = time.perf_counter()
    experiment_id = f"debate_async_{uuid.uuid4().hex[:6]}"

    agents = [
        ("planner", agent_planner),
        ("engineer", agent_engineer),
        ("skeptic", agent_skeptic),
        ("ethicist", agent_ethicist),
    ]

    all_responses = []
    context = ""

    print("=" * 70)
    print(f"ASYNC DEBATE STARTING: '{topic}'")
    print(f"Rounds: {rounds} | Agents: {len(agents)}")
    print("=" * 70)

    for round_num in range(1, rounds + 1):
        round_start = time.perf_counter()
        print(f"\n--- Round {round_num} (starting at {round_start - start_time:.2f}s) ---")

        tasks = [
            asyncio.create_task(
                process_agent(name, func, topic, context, round_num,
                              experiment_id, filename, all_responses)
            )
            for name, func in agents
        ]
        await asyncio.gather(*tasks)

        # Build context from this round's responses (in agent list order)
        for name, _ in agents:
            entry = next(r for r in all_responses
                         if r["round"] == round_num and r["agent"] == name)
            context += f"Round {round_num} - {name}: {entry['response']}\n\n"

        round_elapsed = time.perf_counter() - round_start
        print(f"--- Round {round_num} completed in {round_elapsed:.2f}s ---")

    total_time = time.perf_counter() - start_time
    theoretical_min = rounds * 1.0

    print("\n" + "=" * 70)
    print("ASYNC DEBATE COMPLETE")
    print("=" * 70)
    print(f"Total elapsed time:      {total_time:.2f}s")
    print(f"Theoretical minimum:     {theoretical_min:.2f}s")
    print(f"Efficiency:              {theoretical_min / total_time * 100:.1f}%")
    print(f"Experiment ID:           {experiment_id}")
    print("=" * 70)

    return experiment_id, all_responses


# ═══════════════════════════════════════════════════════════════
# SYNCHRONOUS WRAPPER
# ═══════════════════════════════════════════════════════════════

def run_debate_sync(
    topic: str,
    rounds: int = 3,
    filename: str = "experiment_results.csv",
) -> tuple[str, list[dict]]:
    """Synchronous entry point for callers that can't use await."""
    return asyncio.run(run_debate_async(topic, rounds, filename))


# ═══════════════════════════════════════════════════════════════
# LAB MEETING (async version, backward-compatible entry point)
# ═══════════════════════════════════════════════════════════════

def run_lab_meeting(topic: str, context: str = "") -> str:
    async def gather_all():
        results = await asyncio.gather(
            agent_planner(topic, context),
            agent_engineer(topic, context),
            agent_skeptic(topic, context),
            agent_imagination(topic, context),
            agent_ethicist(topic, context),
        )
        return "\n\n".join(results)

    return asyncio.run(gather_all())


if __name__ == "__main__":
    asyncio.run(run_debate_async(
        topic="Should we implement a four-day work week?",
        rounds=3,
    ))