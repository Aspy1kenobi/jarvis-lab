"""
Prompt construction helpers for each debate agent persona.
"""

SYSTEM_PROMPTS = {
    "skeptic": (
        "You are a rigorous skeptic in a multi-agent debate. "
        "Your job is to surface hidden assumptions, failure modes, and gaps in evidence. "
        "Be concise, specific, and constructive — not contrarian for its own sake."
    ),
    "planner": (
        "You are a strategic planner in a multi-agent debate. "
        "Break problems into clear steps and define what success looks like."
    ),
    "engineer": (
        "You are a pragmatic engineer in a multi-agent debate. "
        "Focus on concrete implementation details, prototypes, and testability."
    ),
    "ethicist": (
        "You are an ethics and safety reviewer in a multi-agent debate. "
        "Identify risks, privacy concerns, and recommend human oversight where needed."
    ),
}


def build_messages(
    agent_name: str,
    topic: str,
    context: str = "",
) -> list[dict]:
    """
    Build the messages list for a given agent, topic, and accumulated context.

    Returns:
        A list of message dicts ready to pass to call_ollama.
    """
    system_prompt = SYSTEM_PROMPTS.get(
        agent_name,
        "You are a helpful assistant in a multi-agent debate.",
    )

    user_content = f"Topic: {topic}"
    if context:
        user_content += f"\n\nContext from previous rounds:\n{context}"
    user_content += "\n\nPlease give your analysis."

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]