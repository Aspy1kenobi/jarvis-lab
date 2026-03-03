import uuid
from agents import agent_planner, agent_engineer, agent_skeptic, agent_ethicist
from experiment_logger import log_result
from scorer import score_response


def run_debate(topic: str, rounds: int = 3, filename: str = "experiment_results.csv") -> tuple[str, list[dict]]:
    """
    Run a multi-agent debate on a given topic for specified number of rounds.

    Args:
        topic: The debate topic/question
        rounds: Number of debate rounds to run (default: 3)
        filename: CSV file to log results to (default: "experiment_results.csv")

    Returns:
        Tuple of (experiment_id, list of dictionaries containing round, agent, and response)
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
                round_num=round_num,
                quality_score=quality_score,
                phase="structured_debate",  # Using string instead of int to match logger expectations
                filename=filename  # Pass the filename through
            )

            all_responses.append({
                "round": round_num,
                "agent": agent_name,
                "response": response,
            })

            context += f"Round {round_num} - {agent_name}: {response}\n\n"

            print(f"{agent_name}: {response[:100]}...")
            print(f"  Quality score: {quality_score:.3f}")

    return experiment_id, all_responses


if __name__ == "__main__":
    experiment_id, results = run_debate("Should we colonize Mars?", rounds=2)

    print("\n" + "=" * 50)
    print("DEBATE SUMMARY")
    print("=" * 50)
    for entry in results:
        print(f"Round {entry['round']} - {entry['agent']}:")
        print(f"{entry['response']}\n")
    
    print(f"\nExperiment ID: {experiment_id}")