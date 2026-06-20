import pandas as pd
import os

# %%
# Define file paths
input_file = 'CollecTF_Output_Reports/40_Threshold_OUTPUT_REPORT/40_raw_report_data.xlsx'
output_dir = 'CollecTF_Output_Reports/40_Threshold_OUTPUT_REPORT'
output_file = os.path.join(output_dir, '40_best_result_all_fields.xlsx')

# # Ensure the output directory exists
# os.makedirs(output_dir, exist_ok=True)

# Load the data
df = pd.read_excel(input_file)

# Sort by p_value to ensure the first record is the lowest
df_sorted = df.sort_values(by='p_value', ascending=True)

# Group by the specified columns and pick the first record (lowest p-value)
# reset_index() keeps the dataframe structure clean
best_results = df_sorted.groupby(['Species Protein', 'UniProt ID']).first().reset_index()

# Save to Excel
best_results.to_excel(output_file, index=False)

print(f"Successfully saved best results to {output_file}")



# Assuming 'best_results' is the dataframe from the previous step
# Group by 'Prospective Pattern', select the 'Alignment' column, and count occurrences
alignment_tally = best_results.groupby('Prospective Pattern')['Alignment'].value_counts()

# To display or save the result
print(alignment_tally)


# Tally the Alignment field across the groups of Prospective Pattern
# normalize=True gives the relative frequency (percentage as a decimal)
# Multiplying by 100 converts it to a standard percentage
alignment_tally_pct = best_results.groupby('Prospective Pattern')['Alignment'].value_counts(normalize=True) * 100

# Convert to DataFrame for easier viewing and saving
alignment_tally_df = alignment_tally_pct.reset_index(name='Percentage')