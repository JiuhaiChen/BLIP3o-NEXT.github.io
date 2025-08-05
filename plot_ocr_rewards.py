#!/usr/bin/env python3
"""
Plot OCR GRPO training rewards from CSV file.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_ocr_rewards():
    """Read CSV and plot OCR training rewards."""
    
    # Set up the plot style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Read the CSV file
    csv_file = 'plot/ocr.csv'
    
    try:
        df = pd.read_csv(csv_file)
        print(f"Successfully loaded CSV with shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
    except FileNotFoundError:
        print(f"Error: Could not find {csv_file}")
        print("Please make sure the CSV file is in the current directory.")
        return
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    
    # Define column names
    x_col = 'train/global_step'
    y_col = 'Qwen3-1.7B-GRPO-16-beta-0.001-lr-1e-6-final-4-nodes - train/rewards/reward_len/mean'
    
    # Check if columns exist
    if x_col not in df.columns:
        print(f"Error: Column '{x_col}' not found in CSV")
        print(f"Available columns: {list(df.columns)}")
        return
        
    if y_col not in df.columns:
        print(f"Error: Column '{y_col}' not found in CSV")
        print("Looking for similar columns...")
        similar_cols = [col for col in df.columns if 'reward' in col.lower() or 'mean' in col.lower()]
        print(f"Columns containing 'reward' or 'mean': {similar_cols}")
        return
    
    # Filter out NaN values and limit to training steps <= 900
    data = df[[x_col, y_col]].dropna()
    data = data[data[x_col] <= 900]
    
    if data.empty:
        print("Error: No valid data points found after removing NaN values")
        return
    
    print(f"Plotting {len(data)} data points")
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot the line with markers
    plt.plot(data[x_col], data[y_col], 
             linewidth=2.5, 
             marker='o', 
             markersize=4, 
             alpha=0.8,
             color='#2E86AB',
             markerfacecolor='#A23B72',
             markeredgecolor='white',
             markeredgewidth=0.5)
    
    # Customize the plot
    plt.title('OCR GRPO Training Reward', 
              fontsize=18, 
              fontweight='bold', 
              pad=20)
    
    plt.xlabel('Training Steps', fontsize=18, fontweight='semibold')
    plt.ylabel('Reward', fontsize=18, fontweight='semibold')
    
    # Add grid
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Increase tick label font size
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    # Improve layout
    plt.tight_layout()
    
    # Create plot directory if it doesn't exist
    plot_dir = 'plot'
    os.makedirs(plot_dir, exist_ok=True)
    
    # Save the plot
    output_path = os.path.join(plot_dir, 'ocr_train_reward.png')
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
        plot_ocr_rewards()