import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Define the thresholds to iterate over (0, 10, 20... 90)
thresholds = range(0, 100, 10)
results = []

for thresh in thresholds:
    filepath = f"{thresh}_Threshold_OUTPUT_REPORT/{thresh}_report_data.xlsx"
    
    if os.path.exists(filepath):
        df = pd.read_excel(filepath)
        
        # Group by the 4 identifiers to collapse the expanded candidate rows back to unique observations
        # Since Execution Time is identical across expanded rows, .first() safely extracts the singular runtime
        unique_obs = df.groupby(['Family', 'Species Protein', 'UniProt ID', 'Analyzed Direction'])['Execution Time (s)'].first()
        
        # Calculate the average execution time across all unique observations for this threshold
        avg_time = unique_obs.mean()
        
        results.append({
            'Threshold (%)': thresh,
            'Average Execution Time (s)': round(avg_time, 4),
            'Num Observations': len(unique_obs)
        })
    else:
        print(f"Warning: {filepath} not found. Skipping...")

if results:
    # 2. Output the Table
    results_df = pd.DataFrame(results)
    print("=== Average Execution Time by Threshold ===")
    print(results_df.to_string(index=False))
    results_df.to_csv("threshold_execution_times.csv", index=False)
    
    # 3. Output the Chart
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=results_df, x='Threshold (%)', y='Average Execution Time (s)', marker='o', linewidth=2, markersize=8, color='teal')
    plt.title('Average Algorithm Execution Time vs. Threshold Percentile', fontsize=14)
    plt.xlabel('Threshold (%)', fontsize=12)
    plt.ylabel('Average Execution Time (seconds)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(list(thresholds))
    plt.tight_layout()
    plt.savefig("Threshold_Execution_Time_Chart.png", dpi=300)
    print("\nAnalysis complete! Saved 'threshold_execution_times.csv' and 'Threshold_Execution_Time_Chart.png'.")
else:
    print("No valid reports were found to analyze.")