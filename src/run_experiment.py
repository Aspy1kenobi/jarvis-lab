import pandas as pd
import os
from debate import run_debate
from experiment_plots import plot_agent_participation as plot_scores_by_agent, plot_quality_curve as plot_scores_over_rounds


def run_experiment(topic: str, rounds: int = 3, filename: str = "experiment_results.csv") -> None:
    """
    Run a complete experiment: debate, logging, analysis, and visualization.
    
    Args:
        topic: The debate topic/question
        rounds: Number of debate rounds to run (default: 3)
        filename: CSV file to log results to (default: "experiment_results.csv")
    """
    # Print header
    print("=" * 70)
    print(f"EXPERIMENT STARTING: Multi-Agent Debate on '{topic}'")
    print(f"Rounds: {rounds}")
    print(f"Output file: {filename}")
    print("=" * 70)
    print()
    
    # Run the debate with explicit filename
    experiment_id, responses = run_debate(topic, rounds, filename=filename)
    
    # Check if CSV exists and load data
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        
        # Filter for this experiment's ID
        experiment_df = df[df['experiment_id'] == experiment_id]
        
        if not experiment_df.empty:
            # Calculate average scores by agent
            print("\n" + "-" * 50)
            print("QUALITY SCORES SUMMARY BY AGENT")
            print("-" * 50)
            
            avg_scores = experiment_df.groupby('agent')['quality_score'].mean().round(3)
            
            # Create a nice table format
            print(f"{'Agent':<15} {'Average Score':<15}")
            print(f"{'-'*15} {'-'*15}")
            for agent, score in avg_scores.items():
                print(f"{agent:<15} {score:<15.3f}")
            
            # Also show overall statistics
            print(f"\nOverall Statistics:")
            print(f"  Total responses: {len(experiment_df)}")
            print(f"  Overall average: {experiment_df['quality_score'].mean():.3f}")
            print(f"  Best agent: {avg_scores.idxmax()} ({avg_scores.max():.3f})")
            print(f"  Worst agent: {avg_scores.idxmin()} ({avg_scores.min():.3f})")
            
            # Call plot functions
            print("\n" + "-" * 50)
            print("GENERATING VISUALIZATIONS")
            print("-" * 50)
            
            # Save the dataframe to a temporary CSV for the plot functions
            # (since they expect a CSV path, not a DataFrame)
            temp_csv = "temp_experiment_data.csv"
            experiment_df.to_csv(temp_csv, index=False)
            
            # Call the plot functions with the CSV path
            plot_scores_by_agent(temp_csv)
            plot_scores_over_rounds(temp_csv)
            
            # Clean up temp file
            os.remove(temp_csv)
            
        else:
            print(f"\nWarning: No data found for experiment ID: {experiment_id} in {filename}")
    else:
        print(f"\nWarning: CSV file '{filename}' not found. No data to analyze.")
    
    # Print footer
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"Files written:")
    print(f"  - {filename} (appended with new results)")
    print(f"  - results/agent_participation.png (bar chart)")
    print(f"  - results/quality_curve.png (line plot)")
    print(f"\nExperiment ID: {experiment_id}")
    print("=" * 70)


if __name__ == "__main__":
    # Run a 3-round debate on climate change
    run_experiment(
        topic="Should governments implement carbon taxes to combat climate change?",
        rounds=3,
        filename="experiment_results.csv"  # Explicitly specify the filename
    )