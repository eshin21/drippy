import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import math 
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import re 

# %%


##############################################################################
## CONCORDANCE / CORRECTNESS TRENDS OVER THRESHOLDS
##############################################################################

thresholds = range(0, 100, 10)
results = []

for thresh in thresholds:
    filepath = f"CollecTF_Output_Reports/{thresh}_Threshold_OUTPUT_REPORT/{thresh}_raw_report_data.xlsx"
    
    if os.path.exists(filepath):
        df = pd.read_excel(filepath)
        
        # 1. Aggregate to find the max Inverted_Pval per group
        # Note: 'correctness_%' must be included in the aggregation or index
        # to ensure it is preserved correctly during the pivot/grouping.
        
        # Suggested aggregation method:
        agg_df = df.groupby(['Species Protein', 'UniProt ID', 'Prospective Pattern']).agg({
            'Inverted_Pval': 'max',
            'correctness_%': 'mean' # Or 'first'/'max' depending on your logic
        }).reset_index()
        
        # 2. Store the average correctness for this threshold
        avg_correctness = agg_df['correctness_%'].mean()
        results.append({'threshold': thresh, 'avg_correctness': avg_correctness})

# 3. Create a summary DataFrame for plotting
summary_df = pd.DataFrame(results)

# 4. Plot
plt.figure(figsize=(10, 6))
plt.plot(summary_df['threshold'], summary_df['avg_correctness'], marker='o')
plt.title('Average Concordance by Threshold')
plt.xlabel('Threshold (%)')
plt.ylabel('Mean Concordance (%)')
plt.grid(True)
plt.show()


# %%

##############################################################################
## COMPUTING COST OVER THRESHOLDS
##############################################################################

for thresh in thresholds:
    filepath = f"CollecTF_Output_Reports/{thresh}_Threshold_OUTPUT_REPORT/{thresh}_raw_report_data.xlsx"
    
    if os.path.exists(filepath):
        df = pd.read_excel(filepath)
        
        # 1. Aggregate to find the max Inverted_Pval per group
        # Note: 'execution  time' must be included in the aggregation or index
        # to ensure it is preserved correctly during the pivot/grouping.
        
        # Suggested aggregation method:
        agg_df = df.groupby(['Species Protein', 'UniProt ID', 'Prospective Pattern']).agg({
            'Inverted_Pval': 'max',
            'Execution Time (s)': 'mean' # Or 'first'/'max' depending on your logic
        }).reset_index()
        
        # 2. Store the average correctness for this threshold
        avg_execution = agg_df['Execution Time (s)'].mean()
        results.append({'threshold': thresh, 'avg_execution': avg_execution})

# 3. Create a summary DataFrame for plotting
summary_df = pd.DataFrame(results)

# 4. Plot
plt.figure(figsize=(10, 6))
plt.plot(summary_df['threshold'], summary_df['avg_execution'], marker='o')
plt.title('Average Execution Time by Threshold (s)')
plt.xlabel('Threshold (%)')
plt.ylabel('Mean Execution Time (%)')
plt.grid(True)
plt.show()



######################
# Youden's J and F1 statistic
# at 0.05 p-value cutoff
#####################


thresholds = range(0, 100, 10)
results = []

for thresh in thresholds:
    # Construct filepath
    folder = f"CollecTF_Output_Reports/{thresh}_Threshold_OUTPUT_REPORT"
    filename = f"{thresh}_best_conclusion_pipeline_data.xlsx"
    path = os.path.join(folder, filename)
    
    if not os.path.exists(path):
        print(f"File not found: {path}")
        continue
        
    df = pd.read_excel(path)
    
    # Calculate stats for DR and IR combined or separately
    # Here we sum the counts for both directions to get a global performance
    stats = {'Threshold': thresh}
    for direction in ['DR', 'IR']:
        col = f'Sys_{direction}_Eval (p<=0.05)'
        counts = df[col].value_counts()
        
        tp = counts.get('TP', 0)
        fp = counts.get('FP', 0)
        tn = counts.get('TN', 0)
        fn = counts.get('FN', 0)
        
        # Calculate metrics
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tpr
        
        # Youden's J
        stats[f'{direction}_J'] = tpr - fpr
        # F1 Score
        stats[f'{direction}_F1'] = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
    results.append(stats)

# Convert to DataFrame for plotting
perf_df = pd.DataFrame(results)



############################################################
# Youden's J and F1 Stat Visualization of Performance vs. Threshold 
############################################################

plt.figure(figsize=(12, 7))

# Define styling for DR (Blue)
plt.plot(perf_df['Threshold'], perf_df['DR_J'], color='blue', linestyle='-', label="DR Youden's J (Solid)")
plt.plot(perf_df['Threshold'], perf_df['DR_F1'], color='blue', linestyle='--', label="DR F1-Score (Dash)")

# Define styling for IR (Red)
plt.plot(perf_df['Threshold'], perf_df['IR_J'], color='red', linestyle='-', label="IR Youden's J (Solid)")
plt.plot(perf_df['Threshold'], perf_df['IR_F1'], color='red', linestyle='--', label="IR F1-Score (Dash)")

plt.title("System Performance: DR vs IR Optimization Metrics (p < 0.05) ")
plt.xlabel("Threshold Percentile (%)")
plt.ylabel("Performance Score")
plt.legend(loc='lower left', bbox_to_anchor=(1, 0.5))
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()

# %%
############################################################
 # Youden's J and F1 Stat but with Optimized cutoffs
############################################################


import pandas as pd
import os
import matplotlib.pyplot as plt
import re

thresholds = range(0, 100, 10)
results = [] # Re-initialize to clear previous runs

for thresh in thresholds:
    path = f"CollecTF_Output_Reports/{thresh}_Threshold_OUTPUT_REPORT/{thresh}_best_conclusion_pipeline_data.xlsx"
    if not os.path.exists(path): 
        continue
        
    df = pd.read_excel(path)
    stats = {'Threshold': thresh}
    
    # Identify dynamic columns
    dr_col = [c for c in df.columns if 'Sys_DR_Eval' in c and 'Opt' in c][0]
    ir_col = [c for c in df.columns if 'Sys_IR_Eval' in c and 'Opt' in c][0]
    
    # Extract optimal values
    stats['DR_Opt_Val'] = float(re.search(r"[-+]?\d*\.\d+|\d+", dr_col).group())
    stats['IR_Opt_Val'] = float(re.search(r"[-+]?\d*\.\d+|\d+", ir_col).group())
    
    for direction, col in zip(['DR', 'IR'], [dr_col, ir_col]):
        counts = df[col].value_counts()
        tp, fp, tn, fn = counts.get('TP', 0), counts.get('FP', 0), counts.get('TN', 0), counts.get('FN', 0)
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tpr
        
        stats[f'{direction}_J'] = tpr - fpr
        stats[f'{direction}_F1'] = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
    # FIX: This must be OUTSIDE the 'for direction' loop
    results.append(stats)

perf_df = pd.DataFrame(results)

# SAFETY NET: Ensure exactly one row per threshold, sorted chronologically
perf_df = perf_df.drop_duplicates(subset=['Threshold']).sort_values('Threshold').reset_index(drop=True)

############################################################
# Visualization of Performance vs. Threshold 
############################################################

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

# Plot 1: Performance Metrics
ax1.plot(perf_df['Threshold'], perf_df['DR_J'], color='blue', linestyle='-', label="DR Youden's J")
ax1.plot(perf_df['Threshold'], perf_df['DR_F1'], color='blue', linestyle='--', label="DR F1-Score")
ax1.plot(perf_df['Threshold'], perf_df['IR_J'], color='red', linestyle='-', label="IR Youden's J")
ax1.plot(perf_df['Threshold'], perf_df['IR_F1'], color='red', linestyle='--', label="IR F1-Score")
ax1.set_title("System Performance: DR vs IR Optimization Metrics (using Optimal Significance Thresholds)")
ax1.set_ylabel("Performance Score")
ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax1.grid(True, linestyle=':', alpha=0.6)

# Plot 2: Evolution of Optimal Thresholds
ax2.plot(perf_df['Threshold'], perf_df['DR_Opt_Val'], color='blue', marker='o', linestyle='-', label="Optimal DR Threshold Value")
ax2.plot(perf_df['Threshold'], perf_df['IR_Opt_Val'], color='red', marker='o', linestyle='-', label="Optimal IR Threshold Value")
ax2.set_title("Evolution of Optimal Significance Threshold Values Across Runs")
ax2.set_xlabel("Pipeline Percentile Threshold (%)")
ax2.set_ylabel("Optimal 1-p_value Threshold")
ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax2.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()



# %%

############################################################
# Visualization of Performance vs. Threshold 
############################################################
import matplotlib.pyplot as plt

# 1. Setup a 2x2 figure grid
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
((ax_roc_dr, ax_pr_dr), (ax_roc_ir, ax_pr_ir)) = axes

# 2. Iterate through thresholds and populate subplots
for thresh in range(0, 100, 10):
    subset = master_df[master_df['Threshold'] == thresh]
    
    y_true_dr = (subset['Prospective Pattern'] == 'DR').astype(int)
    y_true_ir = (subset['Prospective Pattern'] == 'IR').astype(int)
    
    # --- DR Calculations ---
    fpr_dr, tpr_dr, _ = roc_curve(y_true_dr, subset['Sys_DR_Score'])
    auc_dr = auc(fpr_dr, tpr_dr)
    prec_dr, rec_dr, _ = precision_recall_curve(y_true_dr, subset['Sys_DR_Score'])
    
    # --- IR Calculations ---
    fpr_ir, tpr_ir, _ = roc_curve(y_true_ir, subset['Sys_IR_Score'])
    auc_ir = auc(fpr_ir, tpr_ir)
    prec_ir, rec_ir, _ = precision_recall_curve(y_true_ir, subset['Sys_IR_Score'])
    
    # --- Plotting ---
    ax_roc_dr.plot(fpr_dr, tpr_dr, label=f'Thresh {thresh}% (AUC={auc_dr:.2f})')
    ax_pr_dr.plot(rec_dr, prec_dr, label=f'Thresh {thresh}%')
    
    ax_roc_ir.plot(fpr_ir, tpr_ir, label=f'Thresh {thresh}% (AUC={auc_ir:.2f})')
    ax_pr_ir.plot(rec_ir, prec_ir, label=f'Thresh {thresh}%')

# 3. Formatting
for ax in axes.flat:
    ax.legend(fontsize='small', loc='best')
    ax.grid(True, linestyle=':', alpha=0.6)

ax_roc_dr.set_title("Direct Repeats: ROC Curves"); ax_pr_dr.set_title("Direct Repeats: PR Curves")
ax_roc_ir.set_title("Inverted Repeats: ROC Curves"); ax_pr_ir.set_title("Inverted Repeats: PR Curves")

plt.tight_layout()
plt.show()
# %%


