#!/usr/bin/env python3
"""
Plot GenEval GRPO training performance from manual data points.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_geneval_performance():
    """Create plot of training steps vs GenEval performance."""
    
    # Set up the plot style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Manual data: training steps 0-400 every 50 steps and corresponding GenEval performance
    training_steps = [0, 50, 100, 150, 200, 250, 300, 350, 400]
    geneval_performance = [82.1, 83.6, 85.4, 87.6, 87.9, 89.4, 89.1, 90.2, 90.2]
    
    print(f"Plotting {len(training_steps)} data points")
    print(f"Training steps: {training_steps}")
    print(f"GenEval performance: {geneval_performance}")
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot the line with markers
    plt.plot(training_steps, geneval_performance, 
             linewidth=2.5, 
             marker='o', 
             markersize=4, 
             alpha=0.8,
             color='#2E86AB',
             markerfacecolor='#A23B72',
             markeredgecolor='white',
             markeredgewidth=0.5)
    
    # Customize the plot
    plt.title('GenEval GRPO Performance', 
              fontsize=18, 
              fontweight='bold', 
              pad=20)
    
    plt.xlabel('Training Steps', fontsize=18, fontweight='semibold')
    plt.ylabel('GenEval Performance (%)', fontsize=18, fontweight='semibold')
    
    # Add grid
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Increase tick label font size
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    # Set x-axis to show all training step points
    plt.xticks(training_steps)
    
    # Set y-axis range to better show the performance improvement
    plt.ylim(80, 92)
    
    # Improve layout
    plt.tight_layout()
    
    # Create plot directory if it doesn't exist
    plot_dir = 'plot'
    os.makedirs(plot_dir, exist_ok=True)
    
    # Save the plot
    output_path = os.path.join(plot_dir, 'geneval_performance.png')
    plt.savefig(output_path, 
                dpi=300, 
                bbox_inches='tight', 
                facecolor='white',
                edgecolor='none')
    
    print(f"Plot saved successfully to: {output_path}")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    # Install required packages if missing
    try:
        import pandas
        import matplotlib
        import seaborn
    except ImportError as e:
        print(f"Missing package: {e}")
        print("Installing required packages...")
        import subprocess
        import sys
        
        packages = ['pandas', 'matplotlib', 'seaborn', 'numpy']
        for package in packages:
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            except subprocess.CalledProcessError:
                print(f"Failed to install {package}. Please install manually.")
                
        print("Packages installed. Please run the script again.")
    else:
        plot_geneval_performance()