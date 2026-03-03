import re

def score_response(response: str, topic: str) -> float:
    """
    Score a response based on length, topic word specificity, and structure keywords.
    
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
        Score based on presence of role-specific keywords.
        Keywords: plan, risk, implement, concern, ethic, assume, verify, feasib
        """
        keywords = ["plan", "risk", "implement", "concern", "ethic", "assume", "verify", "feasib"]
        
        text_lower = text.lower()
        
        # Count how many keywords appear in the text
        matches = 0
        for keyword in keywords:
            # Use word boundary to match whole words or word parts (for stems like "feasib" matching "feasible")
            if re.search(r'\b' + re.escape(keyword) + r'\w*\b', text_lower):
                matches += 1
        
        # Normalize by total number of keywords
        return matches / len(keywords)
    
    # Calculate individual scores
    len_score = length_score(response)
    spec_score = specificity_score(response, topic)
    struct_score = structure_score(response)
    
    # Calculate average
    final_score = (len_score + spec_score + struct_score) / 3.0
    
    return final_score