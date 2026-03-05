"""
Scoring functions for evaluating agent responses in multi-agent debates.

Components:
- Length score: Measures engagement via response length (words). 200+ words = 1.0.
- Specificity score: Measures topic relevance via word overlap with the original topic.
- Structure score: Measures reasoning surface area via discourse markers and questions.

Known limitations:
- Structure scoring favors explicit reasoning markers (however, because, therefore, etc.).
  Models that reason implicitly or use different discourse conventions will be underscored.
- The current implementation assumes English-language markers.
- Stub-era scores (using keyword matching) and LLM-era scores (using this improved metric)
  are NOT directly comparable. Any analysis across phases must account for this shift.
"""

import re

def score_response(response: str, topic: str) -> float:
    """
    Score a response based on length, topic word specificity, and reasoning structure.
    
    Args:
        response: The agent's response text
        topic: The debate topic/question
    
    Returns:
        A float score between 0.0 and 1.0 (average of three normalized components)
    """
    
    def length_score(text: str) -> float:
        """
        Score based on response length (words).
        200+ words = 1.0, 0 words = 0.0, linear scale in between.
        """
        word_count = len(text.split())
        # Cap at 200 words for maximum score
        return min(word_count / 200.0, 1.0)
    
    def specificity_score(text: str, topic_text: str) -> float:
        """
        Score based on how many words from the topic appear in the response.
        Normalized by number of words in topic.
        """
        # Extract words from topic (split on whitespace and remove punctuation)
        topic_words = set(re.findall(r'\b\w+\b', topic_text.lower()))
        
        if not topic_words:
            return 0.0
        
        # Extract words from response
        response_words = set(re.findall(r'\b\w+\b', text.lower()))
        
        # Count matches
        matches = len(topic_words.intersection(response_words))
        
        # Normalize by number of topic words
        return matches / len(topic_words)
    
    def structure_score(text: str) -> float:
        """
        Score based on reasoning surface area:
        - Discourse markers (60% weight): words that signal reasoning (however, because, therefore, etc.)
        - Questions (40% weight): sentences ending with '?' that surface uncertainty
        
        Both components are normalized by word count (per 100 words) to control for response length.
        """
        # Normalize by word count (per 100 words)
        words = text.split()
        word_count = len(words)
        if word_count == 0:
            return 0.0
        
        # Scale factor to normalize to per-100-words
        scale = 100.0 / word_count
        
        # Discourse markers (weight: 0.6)
        discourse_markers = [
            "because", "since", "therefore", "thus", "hence",  # Reasoning/justification
            "however", "although", "but", "yet", "nevertheless",  # Counterarguments
            "if", "unless", "alternatively", "otherwise",  # Conditional/branching
            "consequently", "as a result", "this implies",  # Implications
            "for example", "for instance", "specifically",  # Elaboration
            "in contrast", "on the other hand", "conversely"  # Comparison
        ]
        
        # Count discourse markers (case-insensitive)
        text_lower = text.lower()
        discourse_count = 0
        for marker in discourse_markers:
            # Use word boundaries to match whole phrases
            pattern = r'\b' + re.escape(marker) + r'\b'
            discourse_count += len(re.findall(pattern, text_lower))
        
        # Normalize discourse score: 5 markers per 100 words = 1.0
        discourse_per_100 = discourse_count * scale
        discourse_score = min(discourse_per_100 / 5.0, 1.0)
        
        # Question score (weight: 0.4)
        # Count sentences ending with '?'
        sentences = re.split(r'[.!?]+', text)
        question_count = text.count('?')
        
        # Normalize question score: 2 questions per 100 words = 1.0
        # (Questions are rarer than discourse markers)
        questions_per_100 = question_count * scale
        question_score = min(questions_per_100 / 2.0, 1.0)
        
        # Weighted combination
        final_structure_score = (discourse_score * 0.6) + (question_score * 0.4)
        
        return final_structure_score
    
    # Calculate individual scores
    len_score = length_score(response)
    spec_score = specificity_score(response, topic)
    struct_score = structure_score(response)
    
    # Calculate average
    final_score = (len_score + spec_score + struct_score) / 3.0
    
    return final_score


# Example usage (commented out)
if __name__ == "__main__":
    # Test cases
    topic = "Should we colonize Mars?"
    
    test_responses = [
        # Short, low-reasoning response
        "Yes, we should colonize Mars.",
        
        # Reasoning-rich response with discourse markers and questions
        """We should consider colonizing Mars carefully. However, there are significant risks. 
        For example, radiation exposure could harm colonists. Therefore, we need shielding technology. 
        But is that technically feasible? If we can't solve this, should we proceed? 
        Alternatively, we could start with robotic missions. This implies we need a phased approach.""",
        
        # Question-heavy response (skeptic style)
        """What are the assumptions here? How will we handle emergencies? 
        Is the cost justified? Who benefits from this? What about planetary protection?""",
        
        # Discourse-heavy response (planner style)
        """Because of the technical challenges, we need a phased approach. 
        However, this requires significant funding. Therefore, we should start with robotic missions. 
        If those succeed, then we can consider human missions. Consequently, the timeline extends."""
    ]
    
    print("STRUCTURE SCORE COMPONENT BREAKDOWN")
    print("=" * 50)
    
    for i, response in enumerate(test_responses):
        words = len(response.split())
        
        # Manual calculation for demonstration
        text_lower = response.lower()
        discourse_markers = ["because", "since", "therefore", "however", "although", 
                           "but", "if", "unless", "alternatively", "consequently"]
        discourse_count = sum(1 for m in discourse_markers if m in text_lower)
        
        sentences = re.split(r'[.!?]+', response)
        question_count = sum(1 for s in sentences if '?' in s)
        
        struct_score = structure_score(response)
        
        print(f"\nResponse {i+1} ({words} words):")
        print(f"  Discourse markers: {discourse_count}")
        print(f"  Questions: {question_count}")
        print(f"  Structure score: {struct_score:.3f}")
        print(f"  Response preview: {response[:100]}...")