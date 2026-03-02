import pandas as pd
import os
from datetime import datetime

def log_result(experiment_id, agent, round_num, quality_score, phase, filename="experiment_results.csv"):
    """
    Log one agent response to a CSV file.
    
    Parameters:
    - experiment_id: str — "exp_001", etc.
    - agent: str — "planner", "skeptic", etc.
    - round_num: int — debate round number
    - quality_score: float — 0.0 to 1.0
    - phase: str — "structured_debate", "emergent", etc.
    - filename: str — path to CSV file (default: "experiment_results.csv")
    """
    
    # Create timestamp
    timestamp = datetime.now().isoformat()
    
    # Create a DataFrame with the new row
    new_row = pd.DataFrame([{
        "experiment_id": experiment_id,
        "agent": agent,
        "round": round_num,
        "quality_score": quality_score,
        "phase": phase,
        "timestamp": timestamp
    }])
    
    # Check if file exists
    file_exists = os.path.isfile(filename)
    
    # Append to CSV (write header only if file doesn't exist)
    new_row.to_csv(
        filename, 
        mode='a', 
        header=not file_exists,
        index=False
    )