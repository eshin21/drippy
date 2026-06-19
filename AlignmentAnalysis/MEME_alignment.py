import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import re

thresholds = range(0, 100, 10)
summary_results = []
all_best_conclusion_data = []

# %% 

os.chdir('/Users/enoch/Documents/Local Erill Research/drippy')
for thresh in thresholds:
    thresh_metrics = {'Threshold': thresh}
    
    # Paths
    raw_path = f"MEME_Output_Reports/MEME_{thresh}_Threshold_OUTPUT_REPORT/{thresh}_raw_report_data.xlsx"
    best_path = f"MEME_Output_Reports/MEME_{thresh}_Threshold_OUTPUT_REPORT/{thresh}_best_conclusion_pipeline_data.xlsx"
    
    # ---------------------------------------------------------
    # 1. Process Raw Report Data (Concordance & Execution)
    # ---------------------------------------------------------
    if os.path.exists(raw_path):
        df_raw = pd.read_excel(raw_path)
        

        # Aggregate to find max Inverted_Pval per group to avoid duplicates
        agg_df = df_raw.groupby(['MEME Folder', 'Motif Number', 'ManualAnnotation']).agg({
            'Inverted_Pval': 'max',
            'correctness_%': 'mean',
            'Execution Time (s)': 'mean' 
        }).reset_index()
        
        thresh_metrics['Avg_Concordance_%'] = agg_df['correctness_%'].mean()
        thresh_metrics['Avg_Execution_Time_s'] = agg_df['Execution Time (s)'].mean()
    else:
        print(f"Missing raw data for threshold {thresh}")
        continue

    # ---------------------------------------------------------
    # 2. Process Best Conclusion Data (Metrics & Opt Values)
    # ---------------------------------------------------------
    if os.path.exists(best_path):
        df_best = pd.read_excel(best_path)
        
        # Append for the ROC/PR master dataframe later
        df_best_copy = df_best.copy()
        df_best_copy['Threshold'] = thresh
        all_best_conclusion_data.append(df_best_copy)
        
        # Identify dynamic columns for Optimal Thresholds
        dr_opt_col = [c for c in df_best.columns if 'Sys_DR_Eval' in c and 'Opt' in c][0]
        ir_opt_col = [c for c in df_best.columns if 'Sys_IR_Eval' in c and 'Opt' in c][0]
        
        # Extract optimal values
        thresh_metrics['DR_Optimal_Significance_Thresh'] = float(re.search(r"[-+]?\d*\.\d+|\d+", dr_opt_col).group())
        thresh_metrics['IR_Optimal_Significance_Thresh'] = float(re.search(r"[-+]?\d*\.\d+|\d+", ir_opt_col).group())
        
        # Calculate J and F1 for both standard (p<=0.05) and optimal columns
        eval_targets = {
            'p05': {'DR': f'Sys_DR_Eval (p<=0.05)', 'IR': f'Sys_IR_Eval (p<=0.05)'},
            'opt': {'DR': dr_opt_col, 'IR': ir_opt_col}
        }
        
        for eval_type, cols in eval_targets.items():
            for direction, col_name in cols.items():
                counts = df_best[col_name].value_counts()
                tp = counts.get('TP', 0)
                fp = counts.get('FP', 0)
                tn = counts.get('TN', 0)
                fn = counts.get('FN', 0)
                
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tpr
                
                prefix = f"{direction}_{eval_type}"
                thresh_metrics[f"{prefix}_J"] = tpr - fpr
                thresh_metrics[f"{prefix}_F1"] = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    else:
        print(f"Missing best conclusion data for threshold {thresh}")
        continue
        
    summary_results.append(thresh_metrics)

# ---------------------------------------------------------
# 3. Finalizing Master DataFrames
# ---------------------------------------------------------
summary_df = pd.DataFrame(summary_results)
master_df = pd.concat(all_best_conclusion_data, ignore_index=True)

# Display the summary DataFrame in console
print("\n=== Comprehensive Summary DataFrame ===")
print(summary_df.to_string(index=False))


# ---------------------------------------------------------
# 4. Visualizations
# ---------------------------------------------------------

# Plot 1: Concordance & Computing Cost
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(summary_df['Threshold'], summary_df['Avg_Concordance_%'], marker='o', color='green')
ax1.set_title('Average Concordance by Threshold')
ax1.set_xlabel('Threshold (%)')
ax1.set_ylabel('Mean Concordance (%)')
ax1.grid(True, linestyle=':', alpha=0.6)

ax2.plot(summary_df['Threshold'], summary_df['Avg_Execution_Time_s'], marker='o', color='purple')
ax2.set_title('Average Execution Time by Threshold')
ax2.set_xlabel('Threshold (%)')
ax2.set_ylabel('Mean Execution Time (s)')
ax2.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()


# Plot 2: J and F1 at p<=0.05
plt.figure(figsize=(10, 5))
plt.plot(summary_df['Threshold'], summary_df['DR_p05_J'], color='blue', linestyle='-', label="DR Youden's J")
plt.plot(summary_df['Threshold'], summary_df['DR_p05_F1'], color='blue', linestyle='--', label="DR F1-Score")
plt.plot(summary_df['Threshold'], summary_df['IR_p05_J'], color='red', linestyle='-', label="IR Youden's J")
plt.plot(summary_df['Threshold'], summary_df['IR_p05_F1'], color='red', linestyle='--', label="IR F1-Score")
plt.title("System Performance: DR vs IR Optimization Metrics (p <= 0.05)")
plt.xlabel("Threshold Percentile (%)")
plt.ylabel("Performance Score")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()


# Plot 3: J and F1 at Optimal Thresholds & Evolution of Optimal Values
fig, (ax3, ax4) = plt.subplots(2, 1, figsize=(10, 10))
ax3.plot(summary_df['Threshold'], summary_df['DR_opt_J'], color='blue', linestyle='-', label="DR Youden's J")
ax3.plot(summary_df['Threshold'], summary_df['DR_opt_F1'], color='blue', linestyle='--', label="DR F1-Score")
ax3.plot(summary_df['Threshold'], summary_df['IR_opt_J'], color='red', linestyle='-', label="IR Youden's J")
ax3.plot(summary_df['Threshold'], summary_df['IR_opt_F1'], color='red', linestyle='--', label="IR F1-Score")
ax3.set_title("System Performance: Optimization Metrics (Optimal Significance Thresholds)")
ax3.set_ylabel("Performance Score")
ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax3.grid(True, linestyle=':', alpha=0.6)

ax4.plot(summary_df['Threshold'], summary_df['DR_Optimal_Significance_Thresh'], color='blue', marker='o', label="Optimal DR Threshold Value")
ax4.plot(summary_df['Threshold'], summary_df['IR_Optimal_Significance_Thresh'], color='red', marker='o', label="Optimal IR Threshold Value")
ax4.set_title("Evolution of Optimal Significance Threshold Values")
ax4.set_xlabel("Pipeline Percentile Threshold (%)")
ax4.set_ylabel("Optimal 1-p_value Threshold")
ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax4.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()


# Plot 4: ROC and PR faceted plots
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
((ax_roc_dr, ax_pr_dr), (ax_roc_ir, ax_pr_ir)) = axes

for thresh in range(0, 100, 10):
    subset = master_df[master_df['Threshold'] == thresh]
    y_true_dr = (subset['ManualAnnotation'] == 'DR').astype(int)
    y_true_ir = (subset['ManualAnnotation'] == 'IR').astype(int)
    
    # DR Calculations
    fpr_dr, tpr_dr, _ = roc_curve(y_true_dr, subset['Sys_DR_Score'])
    auc_dr = auc(fpr_dr, tpr_dr)
    prec_dr, rec_dr, _ = precision_recall_curve(y_true_dr, subset['Sys_DR_Score'])
    
    # IR Calculations
    fpr_ir, tpr_ir, _ = roc_curve(y_true_ir, subset['Sys_IR_Score'])
    auc_ir = auc(fpr_ir, tpr_ir)
    prec_ir, rec_ir, _ = precision_recall_curve(y_true_ir, subset['Sys_IR_Score'])
    
    # Plotting
    ax_roc_dr.plot(fpr_dr, tpr_dr, label=f'Thresh {thresh}% (AUC={auc_dr:.2f})')
    ax_pr_dr.plot(rec_dr, prec_dr, label=f'Thresh {thresh}%')
    ax_roc_ir.plot(fpr_ir, tpr_ir, label=f'Thresh {thresh}% (AUC={auc_ir:.2f})')
    ax_pr_ir.plot(rec_ir, prec_ir, label=f'Thresh {thresh}%')

for ax in axes.flat:
    ax.legend(fontsize='small', loc='best')
    ax.grid(True, linestyle=':', alpha=0.6)

ax_roc_dr.set_title("Direct Repeats: ROC Curves")
ax_pr_dr.set_title("Direct Repeats: PR Curves")
ax_roc_ir.set_title("Inverted Repeats: ROC Curves")
ax_pr_ir.set_title("Inverted Repeats: PR Curves")
plt.tight_layout()
plt.show()

# Export summary to Excel for your records
summary_df.to_excel("MEME_Pipeline_Global_Threshold_Summary.xlsx", index=False)