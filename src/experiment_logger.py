import pandas as pd
import os
from datetime import datetime

def log_result(
    experiment_id,
    agent,
    round_num,
    quality_score,
    phase,
    scoring_path: str = "3-component",
    context_window: str = "round_1_no_context",
    execution_mode: str = "sequential",
    filename: str = "experiment_results.csv",
):
    """
    Log one agent response to a CSV file.

    Parameters:
    - experiment_id: str
    - agent: str
    - round_num: int
    - quality_score: float
    - phase: str
    - scoring_path: str   — "3-component" or "4-component"
    - context_window: str — e.g. "round_1_no_context", "rounds_1-2"
    - execution_mode: str — "sequential" or "parallel"
    - filename: str
    """
    timestamp = datetime.now().isoformat()

    new_row = pd.DataFrame([{
        "experiment_id": experiment_id,
        "agent": agent,
        "round": round_num,
        "quality_score": quality_score,
        "phase": phase,
        "scoring_path": scoring_path,
        "context_window": context_window,
        "execution_mode": execution_mode,
        "timestamp": timestamp,
    }])

    file_exists = os.path.isfile(filename)
    new_row.to_csv(filename, mode='a', header=not file_exists, index=False)