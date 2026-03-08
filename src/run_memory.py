"""
Condition C: Memory-enabled run for Phase 2 experiment.

Calls run_debate_async() directly — all memory logic lives in debate_async.py.
This script exists to make the experimental condition explicit and repeatable.

Run AFTER run_control.py (condition B). Same topic, same CSV file.
Compare context_window values in results:
  condition B: control_window_r1-2
  condition C: memory_shared+private
"""

import asyncio
import sys
from debate_async import run_debate_async

if __name__ == "__main__":
    topic = sys.argv[1] if len(sys.argv) > 1 else \
        "Should AI be used in criminal sentencing?"
    asyncio.run(run_debate_async(topic=topic, rounds=3))