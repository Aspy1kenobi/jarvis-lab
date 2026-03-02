import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_quality_curve(csv_path, output_path="results/quality_curve.png"):
    """
    Line plot: average quality score per round.
    X axis: round number
    Y axis: mean quality_score
    """
    df = pd.read_csv(csv_path)
    
    # Calculate mean quality score per round
    round_means = df.groupby("round")["quality_score"].mean().reset_index()
    
    # Create the line plot with markers
    plt.figure(figsize=(10, 6))
    plt.plot(round_means["round"], round_means["quality_score"], 
             marker='o', linestyle='-', linewidth=2, markersize=8)
    
    # Customize the plot
    plt.title("Average Quality Score by Debate Round", fontsize=14, fontweight='bold')
    plt.xlabel("Round", fontsize=12)
    plt.ylabel("Mean Quality Score", fontsize=12)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.xticks(round_means["round"])  # Ensure all round numbers appear
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save and close
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_agent_participation(csv_path, output_path="results/agent_participation.png"):
    """
    Bar chart: average quality score per agent.
    X axis: agent name
    Y axis: mean quality_score
    """
    df = pd.read_csv(csv_path)
    
    # Calculate mean quality score per agent
    agent_means = df.groupby("agent")["quality_score"].mean().reset_index()
    
    # Create the bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(agent_means["agent"], agent_means["quality_score"], 
            color='skyblue', edgecolor='navy', alpha=0.7)
    
    # Customize the plot
    plt.title("Average Quality Score by Agent", fontsize=14, fontweight='bold')
    plt.xlabel("Agent", fontsize=12)
    plt.ylabel("Mean Quality Score", fontsize=12)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on top of bars
    for i, v in enumerate(agent_means["quality_score"]):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save and close
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()