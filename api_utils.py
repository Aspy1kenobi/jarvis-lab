def parse_llm_response(response: dict) -> tuple[str, dict]:
    """
    Parse an Anthropic API response dictionary and extract the text content and usage information.
    
    Args:
        response: The API response dictionary from Anthropic
        
    Returns:
        Tuple of (text_string, usage_dict)
        
    Raises:
        ValueError: If content is missing, empty, or malformed
    """
    # Check if response has content
    if "content" not in response:
        raise ValueError("Response missing 'content' field")
    
    if not isinstance(response["content"], list):
        raise ValueError(f"Expected 'content' to be a list, got {type(response['content'])}")
    
    if len(response["content"]) == 0:
        raise ValueError("Response content list is empty")
    
    # Extract text from first content item
    first_content = response["content"][0]
    
    if not isinstance(first_content, dict):
        raise ValueError(f"Expected content item to be a dict, got {type(first_content)}")
    
    if "text" not in first_content:
        raise ValueError("Content item missing 'text' field")
    
    text = first_content["text"]
    
    # Extract usage information (default to empty dict if not present)
    usage = response.get("usage", {})
    
    return text, usage