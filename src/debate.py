import uuid
from agents import agent_planner, agent_engineer, agent_skeptic, agent_ethicist
from experiment_logger import log_result
from scorer import score_response


def run_debate(topic: str, rounds: int = 3) -> list[dict]:
    """
    Run a multi-agent debate on a given topic for specified number of rounds.

    Args:
        topic: The debate topic/question
        rounds: Number of debate rounds to run (default: 3)

    Returns:
        List of dictionaries, each containing round, agent, and response
    """
    experiment_id = f"debate_{uuid.uuid4().hex[:6]}"

    agents = [
        ("planner", agent_planner),
        ("engineer", agent_engineer),
        ("skeptic", agent_skeptic),
        ("ethicist", agent_ethicist),
    ]

    all_responses = []
    context = ""

    for round_num in range(1, rounds + 1):
        print(f"\n--- Round {round_num} ---")

        for agent_name, agent_func in agents:
            response = agent_func(topic, context)
            
            # Calculate quality score using the scorer function
            quality_score = score_response(response, topic)

            log_result(
                experiment_id=experiment_id,
                agent=agent_name,
                round=round_num,
                quality_score=quality_score,  # Now using actual score instead of placeholder
                phase=1,
            )

            all_responses.append({
                "round": round_num,
                "agent": agent_name,
                "response": response,
            })

            context += f"Round {round_num} - {agent_name}: {response}\n\n"

            print(f"{agent_name}: {response[:100]}...")
            print(f"  Quality score: {quality_score:.3f}")  # Optional: print score for visibility

    return all_responses
