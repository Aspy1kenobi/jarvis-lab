import re

def length_score(text: str) -> float:
    word_count = len(text.split())
    return min(word_count / 200.0, 1.0)


def specificity_score(text: str, topic_text: str) -> float:
    topic_words = set(re.findall(r'\b\w+\b', topic_text.lower()))
    if not topic_words:
        return 0.0
    response_words = set(re.findall(r'\b\w+\b', text.lower()))
    matches = len(topic_words.intersection(response_words))
    return matches / len(topic_words)


def structure_score(text: str) -> float:
    keywords = ["plan", "risk", "implement", "concern", "ethic", "assume", "verify", "feasib"]
    text_lower = text.lower()
    matches = sum(
        1 for kw in keywords
        if re.search(r'\b' + re.escape(kw) + r'\w*\b', text_lower)
    )
    return matches / len(keywords)


def engagement_score(text: str, context: str) -> float:
    """
    Score based on how much of the response vocabulary came from prior context.
    Measures whether the agent is building on prior arguments, not just restating
    the original topic.

    Returns 0.0 if context is empty — caller is responsible for conditional exclusion.
    """
    if not context or not text:
        return 0.0

    response_words = set(re.findall(r'\b\w+\b', text.lower()))
    context_words = set(re.findall(r'\b\w+\b', context.lower()))

    if not response_words:
        return 0.0

    shared = response_words & context_words
    return len(shared) / len(response_words)


def score_response(response: str, topic: str, context: str = "") -> tuple[float, str]:
    """
    Score a response across up to four components.
    When context is empty (round 1), engagement is excluded to prevent
    artificial round-1 depression that would confound cross-round comparisons.

    Returns:
        (final_score, scoring_path) where scoring_path is "3-component" or "4-component"
    """
    len_score = length_score(response)
    spec_score = specificity_score(response, topic)
    struct_score = structure_score(response)

    if context:
        eng_score = engagement_score(response, context)
        final_score = (len_score + spec_score + struct_score + eng_score) / 4.0
        scoring_path = "4-component"
    else:
        final_score = (len_score + spec_score + struct_score) / 3.0
        scoring_path = "3-component"

    return final_score, scoring_path