import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Define the thresholds to iterate over (0, 10, 20... 90)
thresholds = range(0, 100, 10)
results = []
evaluation_results = []
all_observations = []

categories = ['Strong agree', 'Weak agree', 'Strong disagree', 'Weak disagree', 'No candidates found']

for thresh in thresholds:
    filepath = f"{thresh}_Threshold_OUTPUT_REPORT/{thresh}_best_conclusion_data.xlsx"
    
    if os.path.exists(filepath):
        df = pd.read_excel(filepath)
        
        total_obs = len(df)
        
        if 'Alignment' in df.columns:
            # Standardize string capitalization just to be safe
            df['Alignment'] = df['Alignment'].astype(str).str.strip().str.capitalize()
            
            counts = df['Alignment'].value_counts()
            
            row = {
                'Threshold (%)': thresh,
                'Total Observations': total_obs
            }
            
            # Calculate proportion for each category
            for cat in categories:
                # Capitalize matches the "Strong agree" format
                count = counts.get(cat.capitalize(), 0)
                row[cat] = count / total_obs if total_obs > 0 else 0
                
            results.append(row)
            
            # Save the raw dataframe data for the trajectory plot
            df_copy = df.copy()
            df_copy['Threshold (%)'] = thresh
            all_observations.append(df_copy)
        else:
            print(f"Warning: 'Alignment' column not found in {filepath}. Skipping...")
            
        if 'evaluation' in df.columns:
            # Safely handle missing/null evaluations as "No candidates found"
            eval_series = df['evaluation'].fillna('No candidates found').astype(str).str.strip()
            eval_series = eval_series.replace(['nan', 'None', ''], 'No candidates found')
            eval_counts = eval_series.value_counts()
            
            eval_row = {
                'Threshold (%)': thresh,
                'Total Observations': total_obs
            }
            
            for cat, count in eval_counts.items():
                eval_row[f"{cat} (Count)"] = count
                eval_row[f"{cat} (Proportion)"] = count / total_obs if total_obs > 0 else 0
                
            evaluation_results.append(eval_row)
        else:
            print(f"Warning: 'evaluation' column not found in {filepath}. Skipping...")
    else:
        print(f"Warning: {filepath} not found. Skipping...")

if results:
    # 2. Output the Table
    results_df = pd.DataFrame(results)
    print("=== Alignment Proportions by Threshold ===")
    print(results_df.to_string(index=False))
    results_df.to_csv("threshold_alignment_proportions.csv", index=False)
    
    # 3. Output the Chart
    # Melt the dataframe so we can plot multiple lines easily with Seaborn
    melted_df = results_df.melt(
        id_vars=['Threshold (%)', 'Total Observations'],
        value_vars=categories,
        var_name='Alignment Category',
        value_name='Proportion'
    )
    
    plt.figure(figsize=(10, 6))
    
    # Define intuitive color palette for the categories
    palette = {
        'Strong agree': 'darkgreen',
        'Weak agree': 'lightgreen',
        'Strong disagree': 'darkred',
        'Weak disagree': 'lightcoral',
        'No candidates found': 'gray'
    }
    
    sns.lineplot(
        data=melted_df, 
        x='Threshold (%)', 
        y='Proportion', 
        hue='Alignment Category',
        marker='o', 
        linewidth=2, 
        markersize=8,
        palette=palette
    )
    
    plt.title('Proportion of Alignment Categories vs. Threshold Percentile', fontsize=14)
    plt.xlabel('Threshold (%)', fontsize=12)
    plt.ylabel('Proportion of Observations', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(list(thresholds))
    plt.ylim(-0.05, 1.05) # Proportions strictly range from 0 to 1
    
    # Move legend outside the plot area
    plt.legend(title='Alignment Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plt.savefig("Threshold_Alignment_Chart.png", dpi=300)
    print("\nAnalysis complete! Saved 'threshold_alignment_proportions.csv' and 'Threshold_Alignment_Chart.png'.")
else:
    print("No valid reports were found to analyze.")

if evaluation_results:
    # 4. Output the Evaluation Table
    # fillna(0) ensures that if a category doesn't exist at a specific threshold, it defaults to 0 instead of NaN
    eval_df = pd.DataFrame(evaluation_results).fillna(0)
    print("\n=== Evaluation Metrics by Threshold ===")
    print(eval_df.to_string(index=False))
    eval_df.to_csv("threshold_evaluation_metrics.csv", index=False)
    
    # 5. Output the Evaluation Chart
    prop_cols = [col for col in eval_df.columns if col.endswith('(Proportion)')]
    
    if prop_cols:
        eval_melted = eval_df.melt(
            id_vars=['Threshold (%)', 'Total Observations'],
            value_vars=prop_cols,
            var_name='Evaluation Category',
            value_name='Proportion'
        )
        
        # Clean category names for the legend
        eval_melted['Evaluation Category'] = eval_melted['Evaluation Category'].str.replace(' (Proportion)', '', regex=False)
        
        plt.figure(figsize=(10, 6))
        
        sns.lineplot(
            data=eval_melted, 
            x='Threshold (%)', 
            y='Proportion', 
            hue='Evaluation Category',
            marker='s', 
            linewidth=2, 
            markersize=8
        )
        
        plt.title('Proportion of Evaluation Categories vs. Threshold Percentile', fontsize=14)
        plt.xlabel('Threshold (%)', fontsize=12)
        plt.ylabel('Proportion of Observations', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(list(thresholds))
        plt.ylim(-0.05, 1.05) 
        
        plt.legend(title='Evaluation Category', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        plt.savefig("Threshold_Evaluation_Chart.png", dpi=300)
        print("Saved 'Threshold_Evaluation_Chart.png'.")


if all_observations:
    # 6. Observation Trajectories
    obs_df = pd.concat(all_observations, ignore_index=True)
    
    # Compute -log10(p-value). Replace 0 with a very small number to safely avoid -inf errors
    obs_df['-log10(p-value)'] = -np.log10(obs_df['p_value'].astype(float).replace(0, 1e-10))
    
    # Create unique identifier for each observation line
    obs_id_cols = ['Family', 'Species Protein', 'UniProt ID']
    if all(c in obs_df.columns for c in obs_id_cols):
        obs_df['Observation ID'] = obs_df['Family'].astype(str) + " | " + obs_df['Species Protein'].astype(str) + " | " + obs_df['UniProt ID'].astype(str)
    else:
        obs_df['Observation ID'] = obs_df.index.astype(str)
        
    # Clean up alignment for mapping strictly to our palette colors
    obs_df['Alignment'] = obs_df['Alignment'].astype(str).str.strip().str.capitalize()
    obs_df['Alignment'] = obs_df['Alignment'].replace({'Nan': 'No candidates found', 'None': 'No candidates found', '': 'No candidates found'})
    
    # Create consistent jitter for each observation so lines stay straight but spread out
    np.random.seed(42) # For reproducibility
    unique_obs = obs_df['Observation ID'].unique()
    jitter_x = {obs: np.random.uniform(-1.5, 1.5) for obs in unique_obs}
    jitter_y = {obs: np.random.uniform(-0.05, 0.05) for obs in unique_obs}
    
    obs_df['Plot X'] = obs_df['Threshold (%)'] + obs_df['Observation ID'].map(jitter_x)
    obs_df['Plot Y'] = obs_df['-log10(p-value)'] + obs_df['Observation ID'].map(jitter_y)
    
    # Ensure consistent colors for the same observation across subplots
    obs_palette = dict(zip(unique_obs, sns.color_palette('husl', len(unique_obs))))
    
    g = sns.FacetGrid(obs_df, col="Alignment", col_wrap=3, height=4, aspect=1.2, sharey=False)

    g.map_dataframe(
        sns.lineplot,
        x='Plot X',
        y='Plot Y',
        hue='Observation ID',
        units='Observation ID',
        estimator=None,
        palette=obs_palette,
        alpha=0.4,
        linewidth=1.5,
        zorder=1,
        legend=False
    )
    
    g.map_dataframe(
        sns.scatterplot,
        x='Plot X',
        y='Plot Y',
        color='black',
        s=40,
        zorder=2,
        alpha=0.6
    )
    
    g.set_axis_labels('Threshold (%)', '-log10(p-value)')
    g.set_titles(col_template="{col_name}")
    g.set(xticks=list(thresholds))
    
    sig_threshold = -np.log10(0.05)
    
    for ax in g.axes.flat:
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.axhline(sig_threshold, color='red', linestyle='--', linewidth=1.5, alpha=0.7, zorder=0, label='p=0.05 Threshold')
        
    g.fig.subplots_adjust(top=0.88)
    g.fig.suptitle('Observation P-Value Trajectories vs. Threshold Percentile', fontsize=14)
    
    g.savefig("Threshold_Observation_Trajectories.png", dpi=300)
    print("Saved 'Threshold_Observation_Trajectories.png'.")

    # 7. Individual plot for Strong Agree trajectories
    strong_obs_df = obs_df[obs_df['Alignment'] == 'Strong agree'].copy()
    
    if not strong_obs_df.empty:
        plt.figure(figsize=(14, 10))
        
        # Maximize color distinction for just the strong agree observations
        unique_strong_obs = strong_obs_df['Observation ID'].unique()
        strong_palette = dict(zip(unique_strong_obs, sns.color_palette('husl', len(unique_strong_obs))))
        
        sns.lineplot(
            data=strong_obs_df,
            x='Plot X',
            y='Plot Y',
            hue='Observation ID',
            units='Observation ID',
            estimator=None,
            palette=strong_palette,
            alpha=0.6,
            linewidth=2,
            legend=False
        )
        
        sns.scatterplot(
            data=strong_obs_df,
            x='Plot X',
            y='Plot Y',
            hue='Observation ID',
            palette=strong_palette,
            s=80,
            alpha=0.9,
            legend=False
        )
        
        plt.axhline(sig_threshold, color='red', linestyle='--', linewidth=1.5, alpha=0.7, zorder=0, label='p=0.05 Threshold')
        
        plt.title('Observation P-Value Trajectories: Strong Agree Only', fontsize=16)
        plt.xlabel('Threshold (%)', fontsize=14)
        plt.ylabel('-log10(p-value)', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.xticks(list(thresholds))
        
        plt.tight_layout()
        plt.savefig("Strong_Agree_Trajectories.png", dpi=300)
        print("Saved 'Strong_Agree_Trajectories.png'.")
        
        
        # 8. Heatmap for Strong Agree Trajectories
        plt.figure(figsize=(12, 10))
        
        # Pivot data: Rows = Observations, Columns = Thresholds, Values = P-values
        heatmap_data = strong_obs_df.pivot_table(
            index='Observation ID', 
            columns='Threshold (%)', 
            values='-log10(p-value)'
        )
        
        # Sort observations by average significance for a clean visual gradient
        heatmap_data['mean_sig'] = heatmap_data.mean(axis=1)
        heatmap_data = heatmap_data.sort_values('mean_sig', ascending=True).drop(columns=['mean_sig'])
        
        # Plot Heatmap
        sns.heatmap(
            heatmap_data, 
            cmap='viridis',
            cbar_kws={'label': '-log10(p-value)'},
            yticklabels=False, # Change to True if you want to see every single ID on the y-axis
            linewidths=0.5,
            linecolor='black'
        )
        
        plt.title('Heatmap of Strong Agree P-Values Across Thresholds', fontsize=16)
        plt.xlabel('Threshold (%)', fontsize=14)
        plt.ylabel(f'Unique Observations (n={len(heatmap_data)})', fontsize=14)
        
        plt.tight_layout()
        plt.savefig("Strong_Agree_Heatmap.png", dpi=300)
        print("Saved 'Strong_Agree_Heatmap.png'.")
        
        
        # 9. Boxplot of Strong Agree Distributions
        plt.figure(figsize=(12, 8))
        
        sns.boxplot(
            data=strong_obs_df, x='Threshold (%)', y='-log10(p-value)',
            color='lightblue', showfliers=False
        )
        sns.stripplot(
            data=strong_obs_df, x='Threshold (%)', y='-log10(p-value)',
            color='darkblue', alpha=0.6, jitter=True
        )
        
        plt.axhline(sig_threshold, color='red', linestyle='--', linewidth=1.5, alpha=0.7, zorder=0, label='p=0.05 Threshold')
        
        plt.title('Distribution of Strong Agree P-Values Across Thresholds', fontsize=16)
        plt.xlabel('Threshold (%)', fontsize=14)
        plt.ylabel('-log10(p-value)', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig("Strong_Agree_Distributions.png", dpi=300)
        print("Saved 'Strong_Agree_Distributions.png'.")